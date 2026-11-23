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

def load_trained_model(model_dir: Path, vocab_size, device="cpu"):
    """Rebuild the trained language model from its Hydra config and load its weights."""
    
    train_cfg = OmegaConf.load(model_dir / ".hydra" / "config.yaml")
    
    model = instantiate(train_cfg.model, vocab_size=vocab_size)
    
    state = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, train_cfg

def family_targets(x, labels, pad_id):
    """The family label at every real position, IGNORE_INDEX at padding. Shape (B, T).

    The same label at every position teaches the head to predict the family from
    every prefix x_<=i: the noisy predictor that guidance needs.
    """
    targets = labels.unsqueeze(1).expand_as(x).clone()
    targets[x == pad_id] = IGNORE_INDEX
    return targets

@hydra.main(config_path="../../../config/train_classifier_guidance", config_name="default", version_base=None)
def main(cfg: DictConfig):
    
    # Model directory and flag if it does not exist
    model_dir = Path(f"results/models/{cfg.trained_model.name}")
    if not model_dir.exists():
        raise FileNotFoundError(f"No trained model at {model_dir}")

    # Output directory
    out_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Saving outputs to: {os.path.relpath(out_dir)}")
    
    log.info("-"*20 + "CONFIGURATION" + "-"*20 + "\n")
    log.info(OmegaConf.to_yaml(cfg))


    # Device to use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device being used for the run: {device} \n")
    
    # Setting the seed for the run
    # Default is 42 in the ~/config/default.yaml
    set_seed(cfg.seed)

    # Data
    log.info("-"*23 + "DATA" + "-"*23 + "\n")
    train_set = ProteinDataset(cfg.data.train)
    val_set = ProteinDataset(cfg.data.val)
    log.info(train_set)
    log.info(val_set)
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle=False)

    vocab_size = len(train_set.vocab)
    pad_id = train_set.vocab["<PAD>"]
    n_classes = len(train_set.fam_to_idx)
    log.info(f"Vocabulary size: {vocab_size}, classes: {train_set.fam_to_idx}")

    # Loading the transformer model let's go! labubu
    log.info("\n" + "-"*20 + "BASE MODEL (frozen)" + "-"*20 + "\n")
    base_model, base_cfg = load_trained_model(model_dir, vocab_size, device)
    log.info(base_model)
    
    # Load already created Transformer Model
    log.info("\n" +"-"*23 + "MODEL" + "-"*23 + "\n")
    model, train_cfg = load_trained_model(model_dir, vocab_size, device)
    log.info(model)

    # Create the head: input = the transformer's hidden size, output = number of families
    log.info("\n" + "-"*20 + "CLASSIFIER HEAD" + "-"*20 + "\n")
    head = instantiate(cfg.new_model_head,
                       input_features=base_model.embedding_dim,
                       output_features=n_classes)
    log.info(head)

    # Merge them: ClassifierGuidedLM freezes the base model
    model = ClassifierGuidedLM(base_model, head).to(device)

    # Parameters
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log.info(f"Trainable (head): {n_trainable:,}   Frozen (transformer): {n_frozen:,}")

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

    # Loss functions
    log.info("\n" + "-"*20 + "LOSS FUNCTION" + "-"*20 + "\n")
    # mean, for backward()
    loss_fn = instantiate(cfg.loss_fn, ignore_index=IGNORE_INDEX)
    # exact per-position validation loss
    loss_fn_sum = instantiate(cfg.loss_fn, ignore_index=IGNORE_INDEX, reduction="sum")
    log.info(f"{loss_fn}, ignoring index: {IGNORE_INDEX}")

    # Shape check
    log.info("\n" + "-"*20 + "SHAPE CHECK" + "-"*20 + "\n")
    x, _, labels, _ = next(iter(train_loader))
    x, labels = x.to(device), labels.to(device)
    targets = family_targets(x, labels, pad_id)

    with torch.no_grad():
        logits = model(x)
        batch_loss_sum = loss_fn_sum(logits.transpose(1, 2), targets).item()

    B, T = x.shape
    log.info(f"x (inputs):       {tuple(x.shape)}  [batch, seq_len]")
    log.info(f"labels (family):  {tuple(labels.shape)}  [batch]")
    log.info(f"targets:          {tuple(targets.shape)}  [batch, seq_len]")
    log.info(f"logits (output):  {tuple(logits.shape)}  [batch, seq_len, classes]")
    assert logits.shape == (B, T, n_classes), f"expected {(B, T, n_classes)}, got {tuple(logits.shape)}"

    n_real = (targets != IGNORE_INDEX).sum().item()
    log.info(f"Real positions in this batch: {n_real} of {targets.numel()}")
    log.info(f"First batch loss: {batch_loss_sum / n_real:.4f} should be close to ln({n_classes}) = {np.log(n_classes):.4f}")

    # Training
    log.info("\n" + "-"*20 + "TRAINING" + "-"*20 + "\n")
    epochs = cfg.train.epochs
    validation_every_steps = cfg.train.val_every_step

    step = 0
    # ClassifierGuidedLM.forward keeps the base model in eval mode
    model.train()          

    train_loss, val_loss, val_acc, val_acc_full = [], [], [], []
    best_val = float("inf")

    print("Hello World!")

    """for epoch in range(epochs):

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
    plt.close()"""
    

if __name__ == "__main__":
    start = time.time()
    log.info(f"Started:  {datetime.now():%Y-%m-%d %H:%M:%S}")
    main()
    log.info(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"Duration: {(time.time() - start) / 60:.1f} min")