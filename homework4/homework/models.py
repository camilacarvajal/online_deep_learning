#Used generative ai to help with assignment
from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
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

        c2 = 20
        self.fc1 = nn.Linear(2*n_track*2, c2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(c2, c2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(c2, n_waypoints*2)


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
        x = torch.cat([track_left, track_right], dim=1)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)  
        x = x.reshape(x.size(0), self.n_waypoints, 2) # Reshape to (b, n_waypoints, 2)

        return x


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 20,
    ):
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        self.input_projection = nn.Linear(2*n_track*2, d_model) 
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, n_waypoints * 2) 

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
        # Concatenate and flatten track data
        src = torch.cat([track_left, track_right], dim=1).flatten(start_dim=1) 

        # Project to d_model dimensions
        src = self.input_projection(src)

        # Simple feedforward layers (no Transformer)
        output = self.linear1(src)
        output = torch.relu(output)
        output = self.linear2(output)

        # Reshape to (batch_size, n_waypoints, 2)
        output = output.reshape(-1, self.n_waypoints, 2)
        return output


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        cnn_layers = []
        kernel_size = 3

        cnn_layers.append(torch.nn.Conv2d(3, 3, kernel_size, 1, (kernel_size-1)//2))
        cnn_layers.append(torch.nn.ReLU())

        cnn_layers.append(torch.nn.Conv2d(3, n_waypoints, kernel_size, 1, (kernel_size-1)//2))
        cnn_layers.append(torch.nn.ReLU())

        # Add layers to reshape the output to (batch_size, n_waypoints, 2)
        cnn_layers.append(torch.nn.AdaptiveAvgPool2d((1, 1))) # Reduce spatial dimensions to 1x1
        cnn_layers.append(torch.nn.Flatten(start_dim=1)) # Flatten the tensor
        cnn_layers.append(torch.nn.Linear(n_waypoints, n_waypoints * 2)) # Project to n_waypoints * 2
        
        self.network = torch.nn.Sequential(*cnn_layers)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        return self.network(x)


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
