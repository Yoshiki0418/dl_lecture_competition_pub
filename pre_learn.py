import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset, PairedDataset
from src.clip_model import CLIP_model
from src.utils import set_seed
from torchvision import transforms


@hydra.main(version_base=None, config_path="configs", config_name="pre_config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # 画像サイズの統一
    ])
    print("Dataloader")

    train_paire_set = PairedDataset(args.train_data_path, args.train_annotations_file, args.img_dir, transform=transform)
    train_paire_loader = torch.utils.data.DataLoader(train_paire_set, shuffle=True, **loader_args)

    val_paire_set = PairedDataset(args.val_data_path, args.val_annotations_file, args.img_dir, transform=transform)
    val_paire_loader = torch.utils.data.DataLoader(val_paire_set, shuffle=False, **loader_args)

    print("loader finish")
    # ------------------
    #       Model
    # ------------------
    model = CLIP_model(train_paire_set.num_channels).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------  
    max_val_loss = float('inf')
    for epoch in range(args.epochs):
        count = 0
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        for X, images in train_paire_loader:
            count += 1
            print(count)
            X, images = X.to(args.device), images.to(args.device)

            loss = model(X, images)
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

        model.eval()
        print("train finish")
        with torch.no_grad():
            val_losses = []
            for X, images in val_paire_loader:
                X, images = X.to(args.device), images.to(args.device)
                loss = model(X, images)
                val_losses.append(loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch+1}: Avg Validation Loss: {avg_val_loss:.4f}")
   
        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if avg_val_loss < max_val_loss:
            max_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            if args.use_wandb:
                wandb.log({"val_loss": avg_val_loss})

if __name__ == "__main__":
    run()
