from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(nn.Module):
    class Block(torch.nn.Module):
      def __init__(self, in_channels, out_channels, stride):
          super().__init__()
          kernel_size = 3
          padding = (kernel_size - 1) // 2
          self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
          self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
          self.relu = torch.nn.ReLU()
          self.norm1 = torch.nn.BatchNorm2d(out_channels)
          self.norm2 = torch.nn.BatchNorm2d(out_channels)

      def forward(self, x):
          x = self.relu(self.norm1(self.c1(x)))
          x = self.relu(self.norm2(self.c2(x)))
          return x

    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 6,
        n_blocks: int = 4
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        cnn_layers = [
            #first special layer
            torch.nn.Conv2d(3, in_channels, kernel_size=11, stride=2, padding=5),
            torch.nn.ReLU()
        ]
        c1 = in_channels
        for _ in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride = 2))
            c1=c2
        #final layer adds a classifier
        cnn_layers.append(torch.nn.Conv2d(c1, 32, kernel_size=1))
        self.network = torch.nn.Sequential(*cnn_layers)
        self.classifier = nn.Linear(32, num_classes)  # Classifier for classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        output = self.network(x)

        # Reshape the output for classification -
        # using AdaptiveAvgPool2d to take the spatial average
        output = nn.functional.adaptive_avg_pool2d(output, (1, 1))

        # Reshape the output to (b, num_classes)
        output = output.flatten(1)
        output = self.classifier(output) # Use the submodule for classification        return output
        return output
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    class DownBlock(torch.nn.Module):
      def __init__(
          self,
          in_channels: int = 3,
          out_channels: int = 3, 
          stride = 1
        ):
          super().__init__()
          self.network = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding =1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding=1),
            torch.nn.ReLU(),
          )
      def forward(self, x):
          logits = self.network(x)
          return logits

    class UpBlock(torch.nn.Module):
      def __init__(
          self,
          in_channels: int = 3,
          out_channels: int = 3, 
          stride = 1,
          output_padding = 0
        ):
          super().__init__()
          self.network = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size = 3, 
                stride = stride, 
                padding =1, 
                output_padding= output_padding, # Adjust output_padding based on stride
            ), 
          )
      def forward(self, x):
          logits = self.network(x)
          return logits

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        n_blocks: int = 2
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        #A single model that performs segmentation and depth regression
        cnn_layers = [
            #first special layer
            torch.nn.Conv2d(3, in_channels, kernel_size=12, stride=2, padding=5),
            torch.nn.ReLU()
        ]
        c1 = in_channels

        self.down1 = self.DownBlock(3, 16, stride = 1)
        self.down2 = self.DownBlock(16, 32, stride = 2)
        self.up1 = self.UpBlock(32,16, stride = 4, output_padding = 3)
        self.up2 = self.UpBlock(16,16, stride = 1, output_padding = 0)

        self.downs = nn.ModuleList([self.down1, self.down2])
        self.ups = nn.ModuleList([self.up1, self.up2])

        self.depth = nn.Conv2d(16, 1, kernel_size=1)
        self.logits = nn.Conv2d(16, num_classes, kernel_size=1)
        self.network = torch.nn.Sequential(*cnn_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        output = self.network(z)
        
        for down in self.downs:
          output = down(output)
          
        for up in self.ups:
          output = up(output)

        logits = self.logits(output)
        depth = self.depth(output)
        
        return logits, depth


    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
