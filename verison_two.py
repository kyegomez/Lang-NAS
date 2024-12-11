import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


@dataclass
class LayerShape:
    in_features: int
    out_features: int
    spatial_dims: Optional[Tuple[int, ...]] = None

    def __str__(self) -> str:
        base = f"in={self.in_features}, out={self.out_features}"
        if self.spatial_dims:
            base += f", spatial={self.spatial_dims}"
        return f"LayerShape({base})"


class LayerConfig(NamedTuple):
    type: str
    params: Dict
    shape: LayerShape


class ArchitectureConfig(NamedTuple):
    layers: List[LayerConfig]
    input_shape: LayerShape
    output_shape: LayerShape


class WeightConfig(NamedTuple):
    tensor: Tensor
    initialization: str
    scale: float


class PromptGuidedNAS(nn.Module):
    def __init__(
        self,
        embedding_model: str = "bert-base-uncased",
        max_layers: int = 12,
        hidden_dim: int = 768,
    ):
        super().__init__()
        self.max_layers = max_layers
        self.hidden_dim = hidden_dim

        # Initialize prompt encoder
        self.tokenizer = AutoTokenizer.from_pretrained(
            embedding_model
        )
        self.prompt_encoder = AutoModel.from_pretrained(
            embedding_model
        )

        # Freeze encoder weights
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        # Architecture proposal networks with shape tracking
        self.architecture_backbone = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Specialized heads
        self.layer_count_head = nn.Linear(256, max_layers)
        self.layer_type_head = nn.Linear(256, len(self.LAYER_TYPES))
        self.shape_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Predicts input/output dimensions
        )

        # Layer-specific parameter predictors
        self.conv_predictor = ConvolutionalPredictor(256)
        self.linear_predictor = LinearPredictor(256)
        self.attention_predictor = AttentionPredictor(256)

        self.register_buffer(
            "input_shape_embedding", torch.randn(1, hidden_dim)
        )

    LAYER_TYPES = ["conv2d", "linear", "attention"]

    def encode_prompt(
        self, prompt: str
    ) -> Tuple[Tensor, Dict[str, int]]:
        """Encode prompt and return tensor shapes"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self.prompt_encoder(**inputs)

        embedding = outputs.last_hidden_state[
            :, 0, :
        ]  # [batch_size, hidden_dim]
        shapes = {
            "embedding": list(embedding.shape),
            "hidden": [embedding.shape[0], self.hidden_dim],
        }

        return embedding, shapes

    def propose_architecture(
        self, prompt_embedding: Tensor, input_shape: LayerShape
    ) -> ArchitectureConfig:
        """Generate complete architecture with shape tracking"""
        prompt_embedding.shape[0]

        # Generate base architecture features
        arch_features = self.architecture_backbone(
            prompt_embedding
        )  # [batch_size, 256]

        # Determine number of layers
        layer_logits = self.layer_count_head(
            arch_features
        )  # [batch_size, max_layers]
        num_layers = (
            torch.argmax(layer_logits, dim=1) + 1
        )  # [batch_size]

        current_shape = input_shape
        layers = []

        # Generate each layer's configuration
        for layer_idx in range(num_layers.item()):
            # Predict layer type
            layer_type_logits = self.layer_type_head(
                arch_features
            )  # [batch_size, num_layer_types]
            layer_type_idx = torch.argmax(
                layer_type_logits, dim=1
            )  # [batch_size]
            layer_type = self.LAYER_TYPES[layer_type_idx.item()]

            # Generate layer configuration based on type
            if layer_type == "conv2d":
                config, current_shape = self.conv_predictor(
                    arch_features, current_shape
                )
            elif layer_type == "linear":
                config, current_shape = self.linear_predictor(
                    arch_features, current_shape
                )
            elif layer_type == "attention":
                config, current_shape = self.attention_predictor(
                    arch_features, current_shape
                )
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            layers.append(
                LayerConfig(
                    type=layer_type,
                    params=config,
                    shape=current_shape,
                )
            )

        return ArchitectureConfig(
            layers=layers,
            input_shape=input_shape,
            output_shape=current_shape,
        )

    def generate_weights(
        self, architecture: ArchitectureConfig
    ) -> Dict[str, WeightConfig]:
        """Generate initialized weights for each layer"""
        weights = {}

        for idx, layer_config in enumerate(architecture.layers):
            if layer_config.type == "conv2d":
                weights[f"layer_{idx}"] = self._init_conv_weights(
                    layer_config
                )
            elif layer_config.type == "linear":
                weights[f"layer_{idx}"] = self._init_linear_weights(
                    layer_config
                )
            elif layer_config.type == "attention":
                weights[f"layer_{idx}"] = (
                    self._init_attention_weights(layer_config)
                )

        return weights

    def _init_conv_weights(self, config: LayerConfig) -> WeightConfig:
        weight_shape = (
            config.params["out_channels"],
            config.params["in_channels"],
            config.params["kernel_size"],
            config.params["kernel_size"],
        )

        weight = torch.empty(weight_shape)
        nn.init.kaiming_normal_(
            weight, mode="fan_out", nonlinearity="relu"
        )

        return WeightConfig(
            tensor=weight, initialization="kaiming_normal", scale=1.0
        )

    def _init_linear_weights(
        self, config: LayerConfig
    ) -> WeightConfig:
        weight_shape = (
            config.params["out_features"],
            config.params["in_features"],
        )
        weight = torch.empty(weight_shape)
        nn.init.xavier_uniform_(weight)

        return WeightConfig(
            tensor=weight, initialization="xavier_uniform", scale=1.0
        )

    def _init_attention_weights(
        self, config: LayerConfig
    ) -> WeightConfig:
        d_model = config.params["d_model"]
        weight_shape = (3, d_model, d_model)  # Query, Key, Value
        weight = torch.empty(weight_shape)
        nn.init.xavier_uniform_(weight)

        return WeightConfig(
            tensor=weight,
            initialization="xavier_uniform",
            scale=1.0 / (d_model**0.5),
        )

    def forward(
        self, prompt: str, input_shape: LayerShape
    ) -> Tuple[ArchitectureConfig, Dict[str, WeightConfig]]:
        """End-to-end generation of architecture and weights with shape tracking"""
        prompt_embedding, shape_dict = self.encode_prompt(prompt)
        logger.info(f"Prompt encoded with shapes: {shape_dict}")

        architecture = self.propose_architecture(
            prompt_embedding, input_shape
        )
        logger.info(
            f"Architecture proposed with {len(architecture.layers)} layers"
        )

        weights = self.generate_weights(architecture)
        logger.info("Weights generated for all layers")

        return architecture, weights


class ConvolutionalPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(
                128, 3
            ),  # 3 parameters: out_channels, kernel_size, stride
        )

    def forward(
        self, features: Tensor, input_shape: LayerShape
    ) -> Tuple[Dict, LayerShape]:
        params = self.net(features)  # Shape: [batch_size, 3]

        # Get first item from batch for each parameter
        out_channels = 2 ** (
            4 + torch.clamp(params[0, 0], 0, 4)
        )  # 16 to 256
        kernel_size = (
            2 * torch.clamp(params[0, 1], 1, 3).int().item() + 1
        )  # 3, 5, 7
        stride = (
            torch.clamp(params[0, 2], 1, 2).int().item()
        )  # 1 or 2

        if input_shape.spatial_dims is None:
            raise ValueError(
                "Convolutional layer requires spatial dimensions"
            )

        # Calculate output spatial dimensions
        h, w = input_shape.spatial_dims
        out_h = ((h - kernel_size) // stride) + 1
        out_w = ((w - kernel_size) // stride) + 1

        config = {
            "in_channels": input_shape.in_features,
            "out_channels": int(out_channels.item()),
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": "valid",
        }

        new_shape = LayerShape(
            in_features=int(out_channels.item()),
            out_features=int(out_channels.item()),
            spatial_dims=(out_h, out_w),
        )

        return config, new_shape


class LinearPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Single parameter: out_features
        )

    def forward(
        self, features: Tensor, input_shape: LayerShape
    ) -> Tuple[Dict, LayerShape]:
        params = self.net(features)  # Shape: [batch_size, 1]

        # Get first item from batch
        out_features = 2 ** (
            6 + torch.clamp(params[0, 0], 0, 5)
        )  # 64 to 2048

        total_input_features = input_shape.in_features
        if input_shape.spatial_dims:
            total_input_features *= torch.prod(
                torch.tensor(input_shape.spatial_dims)
            )

        config = {
            "in_features": total_input_features,
            "out_features": int(out_features.item()),
        }

        new_shape = LayerShape(
            in_features=int(out_features.item()),
            out_features=int(out_features.item()),
        )

        return config, new_shape


class AttentionPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # 2 parameters: embed_dim, num_heads
        )

    def forward(
        self, features: Tensor, input_shape: LayerShape
    ) -> Tuple[Dict, LayerShape]:
        params = self.net(features)  # Shape: [batch_size, 2]

        # Get first item from batch for each parameter
        embed_dim = 2 ** (
            5 + torch.clamp(params[0, 0], 0, 5)
        )  # 32 to 1024
        num_heads = 2 ** (
            1 + torch.clamp(params[0, 1], 0, 3)
        )  # 2 to 16

        # Ensure embed_dim is divisible by num_heads
        embed_dim = (
            int(embed_dim.item()) // int(num_heads.item())
        ) * int(num_heads.item())

        config = {
            "embed_dim": embed_dim,  # Changed from d_model to embed_dim
            "num_heads": int(num_heads.item()),
            "dropout": 0.1,
            "batch_first": True,  # Add batch_first parameter
        }

        new_shape = LayerShape(
            in_features=embed_dim, out_features=embed_dim
        )

        return config, new_shape


def create_model_from_prompt(
    prompt: str, input_shape: LayerShape, device: str = "cpu"
) -> Tuple[nn.Module, Dict[str, List[int]]]:
    """Create model with shape tracking"""
    nas = PromptGuidedNAS().to(device)
    architecture, weights = nas(prompt, input_shape)

    # Fix: Create shape tracking dict with single integers instead of trying to convert to list
    shape_tracking = {
        "input": input_shape.in_features,  # Single integer
        "layers": [],
    }

    model = construct_model(architecture, weights)

    # Track shapes through the model
    for layer in architecture.layers:
        shape_tracking["layers"].append(
            {
                "type": layer.type,
                "in": layer.shape.in_features,  # Single integer
                "out": layer.shape.out_features,  # Single integer
                "spatial_dims": (
                    layer.shape.spatial_dims
                    if layer.shape.spatial_dims
                    else None
                ),
            }
        )

    shape_tracking["output"] = (
        architecture.output_shape.out_features
    )  # Single integer

    return model, shape_tracking


# Wrapper class for MultiheadAttention to make it work with nn.Sequential
class AttentionWrapper(nn.Module):
    def __init__(self, attention_layer: nn.MultiheadAttention):
        super().__init__()
        self.attention = attention_layer

    def forward(self, x: Tensor) -> Tensor:
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        # MultiheadAttention expects shape (seq_len, batch, embed_dim) if batch_first=False
        # or (batch, seq_len, embed_dim) if batch_first=True
        if self.attention.batch_first:
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dimension
            elif x.dim() == 3:
                pass  # Already in correct shape
            else:
                x = x.view(
                    x.size(0), -1, x.size(-1)
                )  # Reshape to (batch, seq, embed_dim)
        else:
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add batch dimension in middle
            elif x.dim() == 3:
                x = x.transpose(
                    0, 1
                )  # Transpose to (seq_len, batch, embed_dim)
            else:
                x = x.view(
                    -1, x.size(0), x.size(-1)
                )  # Reshape to (seq, batch, embed_dim)

        # Apply attention. Using the same input for query, key, and value
        attn_output, _ = self.attention(x, x, x)

        # Return output in expected shape
        return (
            attn_output.mean(dim=1)
            if attn_output.dim() == 3
            else attn_output
        )


def construct_model(
    architecture: ArchitectureConfig, weights: Dict[str, WeightConfig]
) -> nn.Module:
    """Construct PyTorch model with initialized weights"""
    layers = []

    for idx, layer_config in enumerate(architecture.layers):
        if layer_config.type == "conv2d":
            layer = nn.Conv2d(**layer_config.params)
            if idx < len(architecture.layers) - 1:
                layers.extend(
                    [
                        layer,
                        nn.ReLU(),
                        nn.BatchNorm2d(
                            layer_config.params["out_channels"]
                        ),
                    ]
                )
            else:
                layers.append(layer)

        elif layer_config.type == "linear":
            # Add flatten layer before linear if we have spatial dimensions
            if layer_config.shape.spatial_dims and idx > 0:
                layers.append(nn.Flatten())
            layer = nn.Linear(**layer_config.params)
            if idx < len(architecture.layers) - 1:
                layers.extend(
                    [
                        layer,
                        nn.ReLU(),
                        nn.LayerNorm(
                            layer_config.params["out_features"]
                        ),
                    ]
                )
            else:
                layers.append(layer)

        elif layer_config.type == "attention":
            layer = nn.MultiheadAttention(**layer_config.params)
            # For attention layers, we need to handle the shape transformation
            attention_wrapper = AttentionWrapper(layer)
            if idx < len(architecture.layers) - 1:
                layers.extend(
                    [
                        attention_wrapper,
                        nn.ReLU(),
                        nn.LayerNorm(
                            layer_config.params["embed_dim"]
                        ),
                    ]
                )
            else:
                layers.append(attention_wrapper)

        else:
            raise ValueError(
                f"Unknown layer type: {layer_config.type}"
            )

        # Initialize weights
        weight_config = weights[f"layer_{idx}"]
        with torch.no_grad():
            if hasattr(layer, "weight"):
                # Ensure weight tensor has correct shape
                if layer.weight.shape == weight_config.tensor.shape:
                    layer.weight.copy_(weight_config.tensor)
                else:
                    logger.warning(
                        f"Weight shape mismatch for layer {idx}. Expected {layer.weight.shape}, got {weight_config.tensor.shape}"
                    )
                    # Initialize with default initialization
                    if isinstance(layer, nn.Conv2d):
                        nn.init.kaiming_normal_(layer.weight)
                    else:
                        nn.init.xavier_uniform_(layer.weight)

    return nn.Sequential(*layers)


# # Example usage showing shape tracking
# if __name__ == "__main__":
#     input_shape = LayerShape(
#         in_features=3,  # RGB channels
#         out_features=3,
#         spatial_dims=(224, 224),  # Standard image size
#     )

#     prompt = """Create a vision model for medical scans"""

#     model, shape_tracking = create_model_from_prompt(
#         prompt, input_shape
#     )

#     # Print shape tracking information
#     # logger.info("Model architecture shapes:")
#     # logger.info(f"Input features: {shape_tracking['input']}")
#     # for idx, layer in enumerate(shape_tracking["layers"]):
#     #     logger.info(f"Layer {idx} ({layer['type']}):")
#     #     logger.info(f"  Input features: {layer['in']}")
#     #     logger.info(f"  Output features: {layer['out']}")
#     #     if layer["spatial_dims"]:
#     #         logger.info(
#     #             f"  Spatial dimensions: {layer['spatial_dims']}"
#     #         )
#     # logger.info(f"Final output features: {shape_tracking['output']}")
#     logger.info(model)


def save_model_artifacts(
    model: nn.Module,
    architecture: Union[ArchitectureConfig, Dict],
    prompt: str,
    save_dir: str = "generated_models",
    model_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Save the model, its architecture, and metadata
    Returns paths to saved files
    """
    os.makedirs(save_dir, exist_ok=True)

    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"nas_model_{timestamp}"

    model_path = os.path.join(save_dir, f"{model_name}.pt")
    config_path = os.path.join(save_dir, f"{model_name}_config.json")

    # Handle both ArchitectureConfig and dict formats
    if isinstance(architecture, dict):
        arch_config = {
            "layers": [
                {
                    "type": layer["type"],
                    "params": {
                        "in_features": layer["in"],
                        "out_features": layer["out"],
                    },
                    "spatial_dims": layer.get("spatial_dims"),
                }
                for layer in architecture["layers"]
            ],
            "input_shape": {
                "in_features": architecture["input"],
                "out_features": architecture["output"],
            },
        }
    else:
        arch_config = {
            "layers": [
                (layer.type, layer.params)
                for layer in architecture.layers
            ],
            "input_shape": {
                "in_features": architecture.input_shape.in_features,
                "out_features": architecture.input_shape.out_features,
                "spatial_dims": architecture.input_shape.spatial_dims,
            },
        }

    # Save model state
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture_config": arch_config,
            "prompt": prompt,
        },
        model_path,
    )

    # Save human-readable config
    config = {
        "prompt": prompt,
        "model_name": model_name,
        "architecture": arch_config,
        "created_at": datetime.now().isoformat(),
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return {"model_path": model_path, "config_path": config_path}


def load_model(model_path: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a saved model and its configuration
    """
    checkpoint = torch.load(model_path)

    # Reconstruct input shape
    input_shape = LayerShape(
        **checkpoint["architecture_config"]["input_shape"]
    )

    # Reconstruct architecture
    layers = []
    for layer_type, params in checkpoint["architecture_config"][
        "layers"
    ]:
        if layer_type == "conv2d":
            layers.append(nn.Conv2d(**params))
        elif layer_type == "linear":
            layers.append(nn.Linear(**params))
        elif layer_type == "attention":
            attention = nn.MultiheadAttention(**params)
            layers.append(AttentionWrapper(attention))

    model = nn.Sequential(*layers)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint


def run_inference(
    model: nn.Module,
    input_data: torch.Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Run inference with the generated model
    """
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        input_data = input_data.to(device)
        output = model(input_data)

    return output


# age with updated create_model_from_prompt
if __name__ == "__main__":
    input_shape = LayerShape(
        in_features=3, out_features=3, spatial_dims=(224, 224)
    )

    prompt = """Create a vision model for xray analysiss"""

    # Create model and get architecture
    model, shape_tracking = create_model_from_prompt(
        prompt, input_shape
    )

    # Save model and configs
    try:
        save_paths = save_model_artifacts(
            model=model,
            architecture=shape_tracking,  # Now handles dict format
            prompt=prompt,
            model_name="medical_ana;yzr",
        )

        logger.info(f"Model saved to: {save_paths['model_path']}")
        logger.info(f"Config saved to: {save_paths['config_path']}")

        # # Test the model
        # dummy_input = torch.randn(1, 3, 224, 224)
        # with torch.no_grad():
        #     output = model(dummy_input)
        #     logger.info(f"Model input shape: {dummy_input.shape}")
        #     logger.info(f"Model output shape: {output.shape}")

        # # Load and verify
        # loaded_model, config = load_model(save_paths['model_path'])
        # logger.info("\nModel successfully loaded!")

    except Exception as e:
        logger.error(f"Error during model saving/loading: {str(e)}")
        raise
