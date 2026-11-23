import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

import logging
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from torch.utils.data import DataLoader
from protgen.data.make_data import ProteinDataset, IGNORE_INDEX
from protgen.models.classifier_guidance_head import ClassifierGuidedLM
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

@hydra.main(config_path="./../../config/train_classifier_guidance_free", config_name="default", version_base=None)
def main(cfg: DictConfig):
    
    # Model direction and output directories printed let's go
    model_dir = Path(f"results/models/{cfg.trained_model.name}")
    out_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Saving outputs to: {os.path.relpath(out_dir)}")

    # Flag if model does not exist
    if not model_dir.exists():
        raise FileNotFoundError(f"No trained model at {model_dir}")
    
    log.info("-"*20 + "CONFIGURATION" + "-"*20 + "\n")
    log.info(OmegaConf.to_yaml(cfg))

    # Device to use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device being used for the run: {device} \n")
    
    # Setting the seed for the run
    # Default is 42 in the ~/config/default.yaml
    set_seed(cfg.seed)

    # Data loading
    # Not loading testing in here to keep testing and training separate
    log.info("-"*23 + "DATA" + "-"*23 + "\n")
    train_set = ProteinDataset(cfg.data.train)
    log.info(train_set)
    val_set = ProteinDataset(cfg.data.val)
    log.info(val_set)
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle= True)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle= False)

    # Load already created Transformer Model
    log.info("\n" +"-"*23 + "MODEL" + "-"*23 + "\n")
    model, train_cfg = load_trained_model(model_dir, vocab_size, device)
    log.info(model)

    # Classifier Head model
    log.info("\n" +"-"*23 + "MODEL" + "-"*23 + "\n")
    vocab_size = len(train_set.vocab)
    log.info(f"Vocabulary size: {vocab_size}")
    model = instantiate(cfg.model, vocab_size=vocab_size).to(device)
    log.info(model)

    # Create the Optimizer
    log.info("\n" +"-"*20 + "OPTIMIZER" + "-"*20 + "\n")
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    log.info(optimizer)

    # Loss functions
    log.info("\n" + "-"*20 + "LOSS FUNCTION" + "-"*20 + "\n")
    # mean, for backward() in training
    loss_fn = instantiate(cfg.loss_fn, ignore_index=IGNORE_INDEX)           
    # for exact per-token validation loss
    loss_fn_sum = instantiate(cfg.loss_fn, ignore_index=IGNORE_INDEX, reduction="sum")
    log.info(f"{loss_fn}, ignoring index: {IGNORE_INDEX}")

    # Shape check (first batch only, untrained model)
    log.info("\n" + "-"*20 + "SHAPE CHECK" + "-"*20 + "\n")
    x, y, _, _ = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        
        logits = model(x)
        # batch_loss_mean = loss_fn(logits.transpose(1, 2), y).item()
        batch_loss_sum = loss_fn_sum(logits.transpose(1, 2), y).item()

    B, T = x.shape
    log.info(f"x (inputs):      {tuple(x.shape)}  [batch, seq_len]")
    log.info(f"y (targets):     {tuple(y.shape)}  [batch, seq_len]")
    log.info(f"logits (output): {tuple(logits.shape)}  [batch, seq_len, vocab]")

    n_real = (y != IGNORE_INDEX).sum().item()
    log.info(f"Real (non-padding) targets in this batch: {n_real} of {y.numel()}")

    # log.info(f"First batch, mean loss:              {batch_loss_mean:.4f}")
    log.info(f"First batch loss (sum / real tokens): {batch_loss_sum / n_real:.4f}")
    
    # assert abs(batch_loss_mean - batch_loss_sum / n_real) < 1e-4, "sum/count does not match mean, check ignore_index"

    log.info(f"Initial loss should be close to ln(vocab) = {np.log(vocab_size):.4f}")

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,}")

    log.info("\n" +"-"*20 + "TRAINING" + "-"*20 + "\n")
    epochs = cfg.train.epochs
    validation_every_steps = cfg.train.val_every_step

    step = 0
    model.train()

    train_loss = []
    val_loss = []
    
    best_val = float("inf")

    for epoch in range(epochs):

        train_loss_batches = []

        for inputs, targets, _, _ in train_loader:          
            
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass.
            output = model(inputs)

            # Compute loss.
            # Keep loss_fn with the mean here. The gradient step should be based on the average loss of the batch.
            loss = loss_fn(output.transpose(1,2), targets)

            train_loss_batches.append(loss.item())

            # Clean up gradients from the model.
            optimizer.zero_grad()

            # Compute gradients based on the loss from the current batch (backpropagation).
            loss.backward()

            # Take one optimizer step using the gradients computed in the previous step.
            optimizer.step()

            step += 1

            if step % validation_every_steps == 0:

                train_loss.append(np.mean(train_loss_batches))
                train_loss_batches = []
                # A mean of batch means is fine here, because it's only used for the curve.

                # Validation: exact per-token loss (sum / count)
                val_total_loss, val_total_tokens = 0.0, 0

                model.eval()
                
                with torch.no_grad():
                    
                    for val_inputs, val_targets, _, _ in val_loader:
                        
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                        
                        val_output = model(val_inputs)
                        
                        # Sum of the loss over all real (non-padding) tokens
                        val_total_loss += loss_fn_sum(val_output.transpose(1, 2), val_targets).item()

                        # How many real tokens
                        val_total_tokens += (val_targets != IGNORE_INDEX).sum().item()

                model.train()

                val_loss.append(val_total_loss / val_total_tokens)

                log.info(f"Step {step:<5}   training loss: {train_loss[-1]:.4f}")
                log.info(f"             validation loss: {val_loss[-1]:.4f}")

                if val_loss[-1] < best_val:
                    best_val = val_loss[-1]
                    torch.save(model.state_dict(), f"{out_dir}/model.pt")
                    log.info(f"  new best val loss {best_val:.4f}, model saved")

    # Outputs
    
    steps = np.arange(1, len(train_loss) + 1) * validation_every_steps
    
    # Output Loss
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump({"steps": steps.tolist(), "train_loss": train_loss,
                   "val_loss": val_loss, "best_val_loss": best_val}, f, indent=2)
    
    # Plot Loss
    plt.figure()
    plt.plot(steps, train_loss, "r", label="Training loss")
    plt.plot(steps, val_loss, "b", label="Validation loss")
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Cross-entropy loss")
    plt.savefig(f"{out_dir}/loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    

if __name__ == "__main__":
    start = time.time()
    log.info(f"Started:  {datetime.now():%Y-%m-%d %H:%M:%S}")
    main()
    log.info(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"Duration: {(time.time() - start) / 60:.1f} min")