import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

import logging
from pathlib import Path
import time
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os  

from torch.utils.data import DataLoader
from protgen.data.make_data import ProteinDataset, IGNORE_INDEX
from protgen.utils import set_seed

log = logging.getLogger(__name__)

def load_trained_model(model_dir: Path, vocab_size, device: str = "cpu"):
    
    # 1. Read the training config saved by Hydra
    train_cfg = OmegaConf.load(model_dir / ".hydra" / "config.yaml")

    # 2. Rebuild the architecture from that config
    model = instantiate(train_cfg.model, vocab_size) 
    
    # 3. Load the weights
    state = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, train_cfg

@hydra.main(config_path="./../../config/test", config_name="default", version_base=None)
def main(cfg: DictConfig):
    
    model_dir = Path(f"results/models/{cfg.trained_model.name}")
    if not model_dir.exists():
        raise FileNotFoundError(f"No trained model at {model_dir}")

    # Device to use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    # Loading Testing only
    log.info("-"*23 + "DATA" + "-"*23 + "\n")
    test_set = ProteinDataset(cfg.data.test)
    log.info(test_set)
    test_loader = DataLoader(test_set, batch_size=cfg.test.batch_size, shuffle= False)
    vocab_size = len(test_set.vocab)
    log.info(f"Vocabulary size: {vocab_size}")

    log.info("\n" +"-"*23 + "MODEL" + "-"*23 + "\n")
    model, train_cfg = load_trained_model(model_dir, vocab_size, device)
    log.info(model)

    # Loss function definition
    log.info("\n" +"-"*20 + "LOSS FUNCTION" + "-"*20 + "\n")
    loss_fn_sum = instantiate(cfg.loss_fn, ignore_index=IGNORE_INDEX, reduction="sum")
    log.info(f"{loss_fn_sum}, ignoring index: {IGNORE_INDEX}")

    # Shape check
    log.info("\n" +"-"*20 + "SHAPE CHECK" + "-"*20 + "\n")
    x, y, _, _ = next(iter(test_loader))
    x, y = x.to(device), y.to(device)
    logits = model(x)
    log.info(f"Shape check x: {x.shape}, y: {y.shape}, output of the model (logits): {logits.shape}")
    total_loss, total_tokens = 0.0, 0
    total_loss += loss_fn_sum(logits.transpose(1, 2), y).item()
    # How many non padding tokens
    total_tokens += (yg != IGNORE_INDEX).sum().item()
    loss = total_loss / total_tokens
    log.info(f"Test Loss: {loss}")
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,}")

    log.info("\n" +"-"*20 + "TESTING" + "-"*20 + "\n")

    test_loss_batches = []
    model.eval()

    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for inputs, targets, _, _ in test_loader:

            # Inputs and targets to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Outputs out of the model
            output = model(inputs)

            # Summing up all the losses summed up
            total_loss += loss_fn_sum(output.transpose(1, 2), targets).item()

            # How many non padding tokens
            total_tokens += (targets != IGNORE_INDEX).sum().item()

    test_loss = total_loss / total_tokens
    log.info(f"Test loss: {test_loss:.4f}")
    
if __name__ == "__main__":
    start = time.time()
    log.info(f"Started:  {datetime.now():%Y-%m-%d %H:%M:%S}")
    main()
    log.info(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"Duration: {(time.time() - start) / 60:.1f} min")