import csv
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.random import f
from tensorflow.python.ops.summary_ops_v2 import threading
import torch
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from .models import load_model, save_model
from .datasets import road_dataset
from .metrics import DetectionMetric

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = road_dataset.load_data("road_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = road_dataset.load_data("road_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    loss_func_depth = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    train_detection = DetectionMetric()

    # training loop
    for epoch in range(num_epoch):
        train_detection.reset()
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for batch in train_data:
            image = batch["image"].to(device)
            depth = batch["depth"].to(device)
            track = batch["track"].to(device)
            logits, pdepth = model(image)
            
            loss_val = loss_func(logits, track)
            pdepth = pdepth.squeeze(1)
            loss_val_depth = loss_func_depth(pdepth, depth)
            
            optimizer.zero_grad()
            totalloss = loss_val + loss_val_depth
            totalloss.backward()
            optimizer.step()
            global_step += 1
            train_detection.add(logits.argmax(1), track, pdepth, depth)

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                image = batch["image"].to(device)
                depth = batch["depth"].to(device)
                track = batch["track"].to(device)
                logits, pdepth = model(image)

                loss_val = loss_func(logits, track)
                
                pdepth = pdepth.squeeze(1)
                
                loss_val_depth = loss_func_depth(pdepth, depth)
                totalloss = loss_val + loss_val_depth
                metrics["val_acc"].append(totalloss)


        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        #returns a dict (accuracy, iou)
        train_epoch = train_detection.compute()
        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f} "
                f"Accuracy={train_epoch['accuracy']:.4f} "
                f"iou={train_epoch['iou']:.4f} "
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    parser.add_argument("--batch_size", type=int, default=128)

    # pass all arguments to train, specify model name here
    train(**vars(parser.parse_args()))

