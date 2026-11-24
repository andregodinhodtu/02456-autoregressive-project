"""
Example run:
uv run python scripts/classifier/guidance/test_classifier_guidance.py \
    trained_models.transformer=transformer/best_transformer \
    trained_models.head=head/head_ffnn
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

def load_trained_transformer(model_dir: Path, vocab_size, device="cpu"):
    """Rebuild the trained language model from its Hydra config and load its weights."""
    
    train_cfg = OmegaConf.load(model_dir / ".hydra" / "config.yaml")
    
    model = instantiate(train_cfg.model, vocab_size=vocab_size)
    
    state = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, train_cfg

def load_trained_head(model_dir: Path, input_features, output_features, device="cpu"):
    """Rebuild the trained classifier head from its Hydra config and load its weights."""

    train_cfg = OmegaConf.load(model_dir / ".hydra" / "config.yaml")

    model = instantiate(train_cfg.head,
                        input_features=input_features,
                        output_features=output_features)

    state = torch.load(model_dir / "head.pt", map_location=device)
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

@hydra.main(config_path="../../../config/test_classifier_guidance", config_name="default", version_base=None)
def main(cfg: DictConfig):
    
    # Model directory and flag if it does not exist for TRANSFORMER
    model_dir_transformer = Path(f"results/models/{cfg.trained_models.transformer}")
    if not model_dir_transformer.exists():
        raise FileNotFoundError(f"No trained model at {model_dir_transformer}")

    # Model directory and flag if it does not exist for HEAD
    model_dir_head = Path(f"results/models/{cfg.trained_models.head}")
    if not model_dir_head.exists():
        raise FileNotFoundError(f"No trained model at {model_dir_head}")
    
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

    # Data loading
    # Loading Testing only
    log.info("-"*23 + "DATA" + "-"*23 + "\n")
    test_set = ProteinDataset(cfg.data.test)
    log.info(test_set)
    test_loader = DataLoader(test_set, batch_size=cfg.test.batch_size, shuffle= False)

    vocab_size = len(test_set.vocab)
    pad_id = test_set.vocab["<PAD>"]
    n_classes = len(test_set.fam_to_idx)
    log.info(f"Vocabulary size: {vocab_size}, classes: {test_set.fam_to_idx}")

    # The head belongs to one specific transformer: check the pairing before loading anything
    head_cfg = OmegaConf.load(model_dir_head / ".hydra" / "config.yaml")
    expected = head_cfg.trained_model.name
    if expected != cfg.trained_models.transformer:
        raise ValueError(
            f"Head '{cfg.trained_models.head}' was trained on '{expected}', "
            f"but you passed trained_models.transformer='{cfg.trained_models.transformer}'. "
            f"Re-run with trained_models.transformer={expected}")

    # Loading the transformer model let's go! labubu
    log.info("\n" + "-"*20 + "BASE MODEL TRANSFORMER" + "-"*20 + "\n")
    base_transformer, base_transformer_cfg = load_trained_transformer(model_dir_transformer, vocab_size, device)
    log.info(base_transformer)

    # Loading the trained HEAD model let's go! labubu
    log.info("\n" + "-"*20 + " HEAD LOADING" + "-"*20 + "\n")
    head, _ = load_trained_head(model_dir_head,
                                input_features=base_transformer.embedding_dim,
                                output_features=n_classes,
                                device=device)
    
    # Merge them: ClassifierGuidedLM freezes the base model
    model = ClassifierGuidedLM(base_transformer, head).to(device)
    model.eval()
    log.info("\n" + "-"*20 + "NEW MODEL" + "-"*20 + "\n")
    log.info(f"New model, check it out below:")
    log.info(model)

    # Parameters
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log.info(f"Trainable (head): {n_trainable:,}   Frozen (transformer): {n_frozen:,}")

    # Loss functions
    log.info("\n" + "-"*20 + "LOSS FUNCTION" + "-"*20 + "\n")
    # mean, for backward()
    loss_fn = instantiate(cfg.loss_fn, ignore_index=IGNORE_INDEX)
    # exact per-position validation loss
    loss_fn_sum = instantiate(cfg.loss_fn, ignore_index=IGNORE_INDEX, reduction="sum")
    log.info(f"{loss_fn}, ignoring index: {IGNORE_INDEX}")

    # Shape check
    log.info("\n" + "-"*20 + "SHAPE CHECK" + "-"*20 + "\n")
    x, _, labels, _ = next(iter(test_loader))
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
    log.info(f"First batch loss: {batch_loss_sum / n_real:.4f}, random would be: ln({n_classes}) = {np.log(n_classes):.4f}")

      # Testing
    log.info("\n" + "-"*20 + "TESTING" + "-"*20 + "\n")

    model.eval()

    # Overall metrics
    total_loss, total_pos, correct = 0.0, 0, 0
    full_correct, full_total = 0, 0
    batch_means = []

    # Per-position metrics
    n_pos        = test_set.max_length
    pos_count    = torch.zeros(n_pos)
    # mean probability per family
    pos_prob_sum = torch.zeros(n_pos, n_classes)
    # mean of the highest probability
    pos_conf_sum = torch.zeros(n_pos)
    # accuracy
    pos_correct  = torch.zeros(n_pos)

    with torch.no_grad():

        for inputs, _, labels, lengths in test_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            targets = family_targets(inputs, labels, pad_id)

            logits = model(inputs)

            # Sum of the loss over all real positions, plus the batch-mean version
            total_loss += loss_fn_sum(logits.transpose(1, 2), targets).item()
            batch_means.append(loss_fn(logits.transpose(1, 2), targets).item())

            # (Batch, Tokens, Class)
            probs = logits.softmax(dim=-1)

            # .max returns both the highest probability and the class it belongs to
            conf, preds = probs.max(dim=-1)

            # Accuracy over all prefix lengths (the noisy predictor)
            mask = targets != IGNORE_INDEX
            correct += (preds[mask] == targets[mask]).sum().item()
            total_pos += mask.sum().item()

            # Accuracy on the complete sequence: the last real position, <EOS>
            last = (lengths + 1).to(device)
            last_preds = preds[torch.arange(len(last), device=device), last]
            full_correct += (last_preds == labels).sum().item()
            full_total += len(last)

            # Real positions only, flattened: padding never enters the statistics
            b, t  = mask.nonzero(as_tuple=True)
            t_cpu = t.cpu()

            pos_count.index_add_(0, t_cpu, torch.ones(len(t_cpu)))
            pos_prob_sum.index_add_(0, t_cpu, probs[b, t].float().cpu())
            pos_conf_sum.index_add_(0, t_cpu, conf[b, t].float().cpu())
            pos_correct.index_add_(0, t_cpu, (preds[b, t] == targets[b, t]).float().cpu())

    test_loss         = total_loss / total_pos
    test_loss_batches = np.mean(batch_means)
    test_acc          = correct / total_pos
    test_acc_full     = full_correct / full_total

    log.info(f"Test loss (per position, sum/count): {test_loss:.4f}")
    log.info(f"Test loss (mean of batch means):     {test_loss_batches:.4f}")
    log.info(f"Difference:                          {test_loss_batches - test_loss:+.4f}")
    log.info(f"An untrained head would give ln({n_classes}) = {np.log(n_classes):.4f}")
    log.info(f"Accuracy (all prefixes):             {test_acc:.4f}")
    log.info(f"Accuracy (full sequence):            {test_acc_full:.4f}")
    log.info(f"Chance level:                        {1 / n_classes:.4f}")

    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump({"test_loss": test_loss,
                   "test_loss_batches": test_loss_batches,
                   "test_acc": test_acc,
                   "test_acc_full": test_acc_full,
                   "n_test": len(test_set),
                   "transformer": cfg.trained_models.transformer,
                   "head": cfg.trained_models.head}, f, indent=2)

    # Per-position results
    log.info("\n" + "-"*20 + "TEST BY POSITION" + "-"*20 + "\n")

    keep = pos_count > 0
    n    = pos_count[keep]

    by_pos = pd.DataFrame({
        "position":   torch.arange(n_pos)[keep].numpy(),
        "n":          n.numpy(),
        "accuracy":   (pos_correct[keep] / n).numpy(),
        "confidence": (pos_conf_sum[keep] / n).numpy(),
    })
    for name, c in test_set.fam_to_idx.items():
        by_pos[f"p_{name}"] = (pos_prob_sum[keep, c] / n).numpy()

    by_pos.to_csv(f"{out_dir}/test_by_position.csv", index=False)
    log.info("\n" + by_pos.head(10).to_string(index=False, float_format="%.4f"))

    chance = 1 / n_classes

    # Cut the x-axis where too few sequences are left to mean anything
    enough = by_pos["n"] >= 0.1 * by_pos["n"].max()
    x_max  = by_pos.loc[enough, "position"].max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)

    # Left: where the probability mass goes, family by family
    for name in test_set.fam_to_idx:
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

    fig.suptitle(f"Family prediction along the sequence, test set  "
                 f"(n = {int(by_pos['n'].iloc[0])} sequences)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/test_by_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    start = time.time()
    log.info(f"Started:  {datetime.now():%Y-%m-%d %H:%M:%S}")
    main()
    log.info(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"Duration: {(time.time() - start) / 60:.1f} min")