import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

import logging
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

@hydra.main(config_path="./../../config/train_classifier_free", config_name="default", version_base=None)
def main(cfg: DictConfig):
    
    out_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Saving outputs to: {os.path.relpath(out_dir)}")
    
    log.info("-"*20 + "CONFIGURATION" + "-"*20 + "\n")
    log.info(OmegaConf.to_yaml(cfg))
    
    # Setting the seed for the run
    # Default is 42 in the ~/config/default.yaml
    set_seed(cfg.seed)

    # Device to use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Device being used for the run: {device} \n")

    # Data loading
    # Not loading testing in here to keep testing and training separate
    log.info("-"*23 + "DATA" + "-"*23 + "\n")
    train_set = ProteinDataset(cfg.data.train)
    log.info(train_set)
    val_set = ProteinDataset(cfg.data.val)
    log.info(val_set)
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle= True)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle= False)

    # Create the model
    log.info("\n" +"-"*23 + "MODEL" + "-"*23 + "\n")
    vocab_size = len(train_set.vocab)
    n_classes = len(train_set.fam_to_idx)
    log.info(f"Vocabulary size: {vocab_size}, classes: {train_set.fam_to_idx}")
    model = instantiate(cfg.model, vocab_size=vocab_size, n_classes=n_classes).to(device)
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
    x, y, labs, _ = next(iter(train_loader))
    x, y, labs = x.to(device), y.to(device), labs.to(device)

    with torch.no_grad():
        logits_cond = model(x, class_labels=labs)
        logits_uncond = model(x)
        # batch_loss_mean = loss_fn(logits.transpose(1, 2), y).item()
        batch_loss_sum = loss_fn_sum(logits_cond.transpose(1, 2), y).item()

    B, T = x.shape
    log.info(f"x (inputs):      {tuple(x.shape)}  [batch, seq_len]")
    log.info(f"y (targets):     {tuple(y.shape)}  [batch, seq_len]")
    log.info(f"labs (class labels):   {tuple(labs.shape)}  [batch]  (min={labs.min().item()}, max={labs.max().item()}, null={model.null_class})")
    log.info(f"logits_cond (output):  {tuple(logits_cond.shape)}  [batch, seq_len, vocab]")
    log.info(f"logits_uncond (output):{tuple(logits_uncond.shape)}  [batch, seq_len, vocab]")

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
    val_loss_cond = []
    val_loss_uncond = []
    
    best_val = float("inf")

    for epoch in range(epochs):

        train_loss_batches = []

        for inputs, targets, labels, _ in train_loader:          
            
            inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device)

            # Randomly masking labels for part of the inputs
            label_mask = torch.rand(labels.shape, device=device) < cfg.train.label_dropout 
            labels[label_mask] = model.null_class 

            # Forward pass.
            output = model(inputs, class_labels=labels)

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

                # Validation: exact per-token loss, both for labeled and unlabeled data (sum / count)
                val_total_loss_cond, val_total_loss_uncond, val_total_tokens = 0.0, 0.0, 0

                model.eval()
                
                with torch.no_grad():
                    
                    for val_inputs, val_targets, val_labels, _ in val_loader:
                        
                        val_inputs, val_targets, val_labels = val_inputs.to(device), val_targets.to(device), val_labels.to(device)
                        
                        val_output_cond = model(val_inputs, class_labels=val_labels)
                        # Sum of the loss over all real (non-padding) tokens
                        val_total_loss_cond += loss_fn_sum(val_output_cond.transpose(1, 2), val_targets).item()

                        # Second validation with unlabeled data as an input
                        val_output_uncond = model(val_inputs, class_labels=None)
                        val_total_loss_uncond += loss_fn_sum(val_output_uncond.transpose(1, 2), val_targets).item()

                        # How many real tokens
                        val_total_tokens += (val_targets != IGNORE_INDEX).sum().item()

                model.train()

                val_loss_cond.append(val_total_loss_cond / val_total_tokens)
                val_loss_uncond.append(val_total_loss_uncond / val_total_tokens)

                log.info(f"Step {step:<5}   training loss: {train_loss[-1]:.4f}")
                log.info(f"             validation loss (cond / uncond): {val_loss_cond[-1]:.4f} / {val_loss_uncond[-1]:.4f}")

                if val_loss_cond[-1] < best_val:
                    best_val = val_loss_cond[-1]
                    torch.save(model.state_dict(), f"{out_dir}/model.pt")
                    log.info(f"  new best val loss {best_val:.4f}, model saved")

    # Outputs
    
    steps = np.arange(1, len(train_loss) + 1) * validation_every_steps
    
    # Output Loss
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump({"steps": steps.tolist(), "train_loss": train_loss,
                   "val_loss_cond": val_loss_cond, "val_loss_uncond": val_loss_uncond,
                     "best_val_loss": best_val}, f, indent=2)
    
    # Plot Loss
    plt.figure()
    plt.plot(steps, train_loss, "r", label="Training loss")
    plt.plot(steps, val_loss_cond, "b", label="Validation loss (labeled data)")
    plt.plot(steps, val_loss_uncond, "g", label="Validation loss (unlabeled data)")
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