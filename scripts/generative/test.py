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

@hydra.main(config_path="./../../config/testing", config_name="default", version_base=None)
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
    log.info(f"Vocabulary size: {vocab_size}")
    model = instantiate(cfg.model, vocab_size=vocab_size).to(device)
    log.info(model)

    # Create the Optimizer
    log.info("\n" +"-"*20 + "OPTIMIZER" + "-"*20 + "\n")
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    log.info(optimizer)

    # Loss function definition
    log.info("\n" +"-"*20 + "LOSS FUNCTION" + "-"*20 + "\n")
    loss_fn = instantiate(cfg.loss_fn, ignore_index=IGNORE_INDEX)
    log.info(f"{loss_fn}, ignoring index: {IGNORE_INDEX}")

    # Shape check
    log.info("\n" +"-"*20 + "SHAPE CHECK" + "-"*20 + "\n")
    x, y, _, _ = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    logits = model(x)
    log.info(f"Shape check x: {x.shape}, y: {y.shape}, output of the model (logits): {logits.shape}")
    loss = loss_fn(logits.transpose(1,2), y)
    log.info(f"Test Loss: {loss}")
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

                val_loss_batches = []

                with torch.no_grad():
                    
                    model.eval()
                    
                    for inputs, targets, _, _ in val_loader:
                        
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        output = model(inputs)
                        
                        loss = loss_fn(output.transpose(1,2), targets)
                        val_loss_batches.append(loss.item())

                    model.train()

                val_loss.append(np.mean(val_loss_batches))

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