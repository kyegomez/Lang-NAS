from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from loguru import logger


class DynamicNAS:
    """
    Dynamic Neural Architecture Search (NAS) and Weight Generation System.

    This system dynamically designs a neural network architecture and generates
    custom weights based on a natural language prompt.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the system.

        Args:
            device (Optional[str]): Specify 'cuda' or 'cpu'. Defaults to 'cuda' if available.
        """
        logger.info("Initializing DynamicNAS...")
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"System running on device: {self.device}")

    def parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Parses the prompt to extract requirements for the architecture.

        Args:
            prompt (str): Natural language description of the task.

        Returns:
            Dict[str, Any]: Parsed information, including model type, input size, etc.
        """
        logger.info(f"Parsing prompt: {prompt}")
        # Example heuristic-based parsing
        if "text" in prompt.lower() or "language" in prompt.lower():
            task = "NLP"
            input_size = (
                512  # Example max sequence length for text tasks
            )
            model_type = "transformer"
        elif "image" in prompt.lower() or "vision" in prompt.lower():
            task = "vision"
            input_size = (3, 64, 64)  # Example image size (C, H, W)
            model_type = "cnn"
        elif (
            "time-series" in prompt.lower()
            or "sequence" in prompt.lower()
        ):
            task = "time-series"
            input_size = (
                128  # Example sequence length for time-series data
            )
            model_type = "lstm"
        else:
            task = "general"
            input_size = 100
            model_type = "mlp"

        output_size = 10  # Default output size (e.g., 10 classes)
        logger.info(
            f"Task: {task}, Model Type: {model_type}, Input Size: {input_size}"
        )
        return {
            "task": task,
            "model_type": model_type,
            "input_size": input_size,
            "output_size": output_size,
        }

    def generate_architecture(
        self, model_type: str, input_size: Any, output_size: int
    ) -> nn.Module:
        """
        Dynamically generates an architecture based on model type and input/output requirements.

        Args:
            model_type (str): The type of model to generate (e.g., transformer, cnn, lstm).
            input_size (Any): Input size specification.
            output_size (int): Output size.

        Returns:
            nn.Module: Generated PyTorch model architecture.
        """
        logger.info(
            f"Generating architecture for model type: {model_type}"
        )
        if model_type == "cnn":
            # Example CNN for vision tasks
            model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 16 * 16, 128),
                nn.ReLU(),
                nn.Linear(128, output_size),
            ).to(self.device)
        elif model_type == "transformer":
            # Example Transformer Encoder
            model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=input_size, nhead=8
                ),
                num_layers=6,
            ).to(self.device)
        elif model_type == "lstm":
            # Example LSTM for sequence tasks
            model = nn.Sequential(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=64,
                    num_layers=2,
                    batch_first=True,
                ),
                nn.Linear(64, output_size),
            ).to(self.device)
        else:  # Default MLP
            model = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_size),
            ).to(self.device)

        logger.info(
            f"Generated {model_type} architecture successfully."
        )
        return model

    def generate_weights(self, model: nn.Module) -> nn.Module:
        """
        Dynamically initializes weights for the given model.

        Args:
            model (nn.Module): Model architecture.

        Returns:
            nn.Module: Model with initialized weights.
        """
        logger.info("Initializing weights for the model...")
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, nonlinearity="relu"
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    layer.weight, nonlinearity="relu"
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LSTM):
                for param in layer.parameters():
                    if param.data.ndimension() >= 2:
                        nn.init.xavier_uniform_(param.data)
                    else:
                        nn.init.zeros_(param.data)
        logger.info("Weights initialized successfully.")
        return model

    def generate_model(self, prompt: str) -> nn.Module:
        """
        Full pipeline: Dynamically generate a model architecture and initialize its weights.

        Args:
            prompt (str): Task description.

        Returns:
            nn.Module: Fully initialized model.
        """
        logger.info(f"Generating model for prompt: {prompt}")

        # Step 1: Parse the prompt
        task_info = self.parse_prompt(prompt)

        # Step 2: Generate architecture
        model = self.generate_architecture(
            model_type=task_info["model_type"],
            input_size=task_info["input_size"],
            output_size=task_info["output_size"],
        )

        # Step 3: Generate weights
        model = self.generate_weights(model)

        logger.info("Model generation complete.")
        return model


# Example Usage
if __name__ == "__main__":
    nas_system = DynamicNAS()
    task_prompt = "Image classification for medical scans"
    try:
        generated_model = nas_system.generate_model(task_prompt)
        logger.info(f"Generated Model:\n{generated_model}")
    except Exception as e:
        logger.error(f"Error during model generation: {e}")
