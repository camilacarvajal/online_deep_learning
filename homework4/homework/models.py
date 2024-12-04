#Used generative ai to help with assignment
from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    class Block(torch.nn.Module):
      def __init__(self, in_channels, out_channels, stride):
          super().__init__()
          kernel_size = 3
          padding = (kernel_size - 1) // 2
          self.c1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
          self.c2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
          self.relu = torch.nn.ReLU()
          self.norm1 = torch.nn.BatchNorm1d(out_channels)
          self.norm2 = torch.nn.BatchNorm1d(out_channels)

      def forward(self, x):
          x = self.relu(self.norm1(self.c1(x)))
          x = self.relu(self.norm2(self.c2(x)))
    
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        n_blocks: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        c1 = 64

        cnn_layers = [
            #first special layer
            torch.nn.Conv1d(20, c1, kernel_size=11, stride=2, padding=5),
            torch.nn.ReLU()
        ]

        for _ in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride = 4))
            c1=c2
        
        cnn_layers.append(torch.nn.Flatten())  # Flatten the output before the linear layer
        linear_input_size = c1 * (n_track * 2) // (2 ** (n_blocks+1)) - 128
        cnn_layers.append(torch.nn.Linear(linear_input_size , 6))
        self.network = torch.nn.Sequential(*cnn_layers)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # optional: normalizes the input
        #z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]


        # Reshape the output for classification -
        concat = torch.cat([track_left, track_right], dim=1)
        
        output = self.network(concat)

        #Reshape the output to (b, -1)
        output = output.view(output.shape[0], -1) 

        output = output.reshape(output.shape[0], self.n_waypoints, 2)  
        return output


class TransformerPlanner(nn.Module):
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
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        cnn_layers = [
            #first special layer
            torch.nn.Conv2d(3, n_track, kernel_size=12, stride=2, padding=5),
            torch.nn.ReLU()
        ]
        c1 = n_track

        self.down1 = self.DownBlock(3, 16, stride = 1)
        self.down2 = self.DownBlock(16, 32, stride = 2)
        self.up1 = self.UpBlock(32,16, stride = 4, output_padding = 3)
        self.up2 = self.UpBlock(16,16, stride = 1, output_padding = 0)

        self.downs = nn.ModuleList([self.down1, self.down2])
        self.ups = nn.ModuleList([self.up1, self.up2])

        self.depth = nn.Conv2d(16, 1, kernel_size=1)
        self.logits = nn.Conv2d(16, n_waypoints, kernel_size=1)
        self.network = torch.nn.Sequential(*cnn_layers)

        self.query_embed = nn.Embedding(n_waypoints, d_model)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        raise NotImplementedError


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
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
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
