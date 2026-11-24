"""
Example sweep:
uv run python scripts/classifier/guidance/train_classifier_guidance.py -m \
    trained_model.name=transformer/best_transformer \
    head.num_layers=0,1,2 \
    head.hidden_units=64,128,256 \
    optimizer.lr=0.003,0.001,0.0003 \
    seed=1,2,3
Example run:
uv run python scripts/classifier/guidance/train_classifier_guidance.py \
    trained_model.name=transformer/best_transformer
"""

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
import pandas as pd
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

    # Create the head: input = the transformer's hidden size, output = number of families
    log.info("\n" + "-"*20 + "CLASSIFIER HEAD" + "-"*20 + "\n")
    head = instantiate(cfg.head,
                       input_features=base_model.embedding_dim,
                       output_features=n_classes)
    log.info(head)

    # Merge them: ClassifierGuidedLM freezes the base model
    model = ClassifierGuidedLM(base_model, head).to(device)
    log.info("\n" + "-"*20 + "NEW MODEL" + "-"*20 + "\n")
    log.info(f"New model, check it out below:")
    log.info(model)

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

    for epoch in range(epochs):

        train_loss_batches = []

        for inputs, _, labels, _ in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            targets = family_targets(inputs, labels, pad_id)

            # Forward pass: frozen transformer -> hidden states -> head
            # (B, T, 3)
            output = model(inputs)

            # Loss over every real position, mean for backward()
            loss = loss_fn(output.transpose(1, 2), targets)

            train_loss_batches.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % validation_every_steps == 0:

                train_loss.append(np.mean(train_loss_batches))
                train_loss_batches = []

                # Validation: exact per-position loss, plus two accuracies
                val_total_loss, val_total_pos, val_correct = 0.0, 0, 0
                full_correct, full_total = 0, 0

                model.eval()

                with torch.no_grad():

                    for val_inputs, _, val_labels, val_lengths in val_loader:

                        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                        val_targets = family_targets(val_inputs, val_labels, pad_id)

                        val_output = model(val_inputs)

                        # Sum of the loss over all real positions
                        val_total_loss += loss_fn_sum(val_output.transpose(1, 2), val_targets).item()

                        # Accuracy over all prefix lengths (the noisy predictor)
                        mask = val_targets != IGNORE_INDEX
                        preds = val_output.argmax(dim=-1)
                        val_correct += (preds[mask] == val_targets[mask]).sum().item()
                        val_total_pos += mask.sum().item()

                        # Accuracy on the complete sequence: the last real position, <EOS>
                        last = (val_lengths + 1).to(device)
                        last_preds = preds[torch.arange(len(last), device=device), last]
                        full_correct += (last_preds == val_labels).sum().item()
                        full_total += len(last)
                
                model.train()
                
                val_loss.append(val_total_loss / val_total_pos)
                val_acc.append(val_correct / val_total_pos)
                val_acc_full.append(full_correct / full_total)

                log.info(f"Step {step:<5}   training loss: {train_loss[-1]:.4f}")
                log.info(f"             validation loss: {val_loss[-1]:.4f}   "
                         f"acc (all prefixes): {val_acc[-1]:.3f}   acc (full seq): {val_acc_full[-1]:.3f}")

                if val_loss[-1] < best_val:
                    best_val = val_loss[-1]
                    
                    # Only the head: the transformer is unchanged in results/models/<name>
                    torch.save(model.classifier_head.state_dict(), f"{out_dir}/head.pt")
                    log.info(f"  new best val loss {best_val:.4f}, head saved")

    # Training curves
    steps = np.arange(1, len(train_loss) + 1) * validation_every_steps

    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump({"steps": steps.tolist(), "train_loss": train_loss, "val_loss": val_loss,
                   "val_acc": val_acc, "val_acc_full": val_acc_full,
                   "best_val_loss": best_val}, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(steps, train_loss, "r", label="Training loss")
    ax1.plot(steps, val_loss, "b", label="Validation loss")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Cross-entropy loss"); ax1.legend()
    ax2.plot(steps, val_acc, "b", label="Val acc (all prefixes)")
    ax2.plot(steps, val_acc_full, "g", label="Val acc (full sequence)")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Accuracy"); ax2.legend()
    fig.savefig(f"{out_dir}/loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-position diagnostics on the best head
    log.info("\n" + "-"*20 + "VALIDATION BY POSITION" + "-"*20 + "\n")
    model.classifier_head.load_state_dict(torch.load(f"{out_dir}/head.pt", map_location=device))
    model.eval()

    n_pos        = train_set.max_length
    
    pos_count    = torch.zeros(n_pos)
    # mean probability per family
    pos_prob_sum = torch.zeros(n_pos, n_classes)
    # mean of the highest probability
    pos_conf_sum = torch.zeros(n_pos)
    # accuracy
    pos_correct  = torch.zeros(n_pos)

    with torch.no_grad():

        for val_inputs, _, val_labels, _ in val_loader:

            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_targets = family_targets(val_inputs, val_labels, pad_id)

            # (Batch, Tokens, Class)
            probs = model(val_inputs).softmax(dim=-1)

            # .max returns both the highest probability and the class it belongs to
            conf, preds = probs.max(dim=-1)

            # Real positions only, flattened: padding never enters the statistics
            b, t  = (val_targets != IGNORE_INDEX).nonzero(as_tuple=True)
            t_cpu = t.cpu()

            pos_count.index_add_(0, t_cpu, torch.ones(len(t_cpu)))
            pos_prob_sum.index_add_(0, t_cpu, probs[b, t].float().cpu())
            pos_conf_sum.index_add_(0, t_cpu, conf[b, t].float().cpu())
            pos_correct.index_add_(0, t_cpu, (preds[b, t] == val_targets[b, t]).float().cpu())

    model.train()

    keep = pos_count > 0
    n    = pos_count[keep]

    by_pos = pd.DataFrame({
        "position":   torch.arange(n_pos)[keep].numpy(),
        "n":          n.numpy(),
        "accuracy":   (pos_correct[keep] / n).numpy(),
        "confidence": (pos_conf_sum[keep] / n).numpy(),
    })
    for name, c in train_set.fam_to_idx.items():
        by_pos[f"p_{name}"] = (pos_prob_sum[keep, c] / n).numpy()

    by_pos.to_csv(f"{out_dir}/val_by_position.csv", index=False)
    log.info("\n" + by_pos.head(10).to_string(index=False, float_format="%.4f"))

    chance = 1 / n_classes

    # Cut the x-axis where too few sequences are left to mean anything
    enough = by_pos["n"] >= 0.1 * by_pos["n"].max()
    x_max  = by_pos.loc[enough, "position"].max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)

    # Left: where the probability mass goes, family by family
    for name in train_set.fam_to_idx:
        ax1.plot(by_pos["position"], by_pos[f"p_{name}"], lw=1.8, label=name)
    ax1.axhline(chance, ls=":", c="grey", lw=1)
    ax1.set_title("Mean predicted probability per family")
    ax1.set_xlabel("Position (tokens seen - 1)")
    ax1.set_ylabel("Probability")
    ax1.legend(title="Family", frameon=False)

    # Right: how sure it is, against how often it is right
    ax2.plot(by_pos["position"], by_pos["confidence"], "k", lw=1.8, label="Confidence (mean max prob)")
    ax2.plot(by_pos["position"], by_pos["accuracy"], "g", lw=1.8, label="Accuracy")
    ax2.axhline(chance, ls=":", c="grey", lw=1, label=f"Chance ({chance:.2f})")
    ax2.fill_between(by_pos["position"], by_pos["confidence"], by_pos["accuracy"],
                     color="grey", alpha=0.15)
    ax2.set_title("Confidence vs accuracy")
    ax2.set_xlabel("Position (tokens seen - 1)")
    ax2.legend(frameon=False)

    for ax in (ax1, ax2):
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Family prediction along the sequence  (n = {int(by_pos['n'].iloc[0])} val sequences)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/val_by_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    start = time.time()
    log.info(f"Started:  {datetime.now():%Y-%m-%d %H:%M:%S}")
    main()
    log.info(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"Duration: {(time.time() - start) / 60:.1f} min")