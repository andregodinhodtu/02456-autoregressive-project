import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

import logging
from pathlib import Path
import time
from datetime import datetime

import torch
import pandas as pd
import os  

from protgen.models.classifier_guidance_head import ClassifierGuidedLM
from protgen.data.make_data import ProteinDataset
from protgen.utils import set_seed


# just started reviewing

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

# generative script for steering
# We can generate without steering as well

@hydra.main(config_path="../../../config/generate_classifier_guidance", config_name="default", version_base=None)
def main(cfg: DictConfig):

    # Model directory and flag if it does not exist for TRANSFORMER
    model_dir_transformer = Path("results/models/transformer") / cfg.trained_models.transformer
    if not model_dir_transformer.exists():
        raise FileNotFoundError(f"No trained model at {model_dir_transformer}")

    # Model directory and flag if it does not exist for HEAD
    model_dir_head = Path("results/models/head") / cfg.trained_models.head
    if not model_dir_head.exists():
        raise FileNotFoundError(f"No trained head at {model_dir_head}")

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
    # Here only to get dynamic variables about the dataset: vocab_size, sequence length, id of the 
    # BOS token and for classifier guidance the classes
    log.info("-"*23 + "DATA" + "-"*23 + "\n")
    train_set = ProteinDataset(cfg.data.train)
    log.info(train_set)
    vocab_size = len(train_set.vocab)
    pad_id = train_set.vocab["<PAD>"]
    n_classes = len(train_set.fam_to_idx)
    log.info(f"Vocabulary size: {vocab_size}, classes: {train_set.fam_to_idx}")

    # Import the models (depending on the config parameters used)
    # The head belongs to one specific transformer: check the pairing before loading anything
    head_cfg = OmegaConf.load(model_dir_head / ".hydra" / "config.yaml")
    # head_cfg is the head's OWN saved config, whose key is singular: leave it as it is
    expected = head_cfg.trained_model.name
    if expected != f"transformer/{cfg.trained_models.transformer}":
        raise ValueError(
            f"Head '{cfg.trained_models.head}' was trained on '{expected}', "
            f"but you passed trained_models.transformer='{cfg.trained_models.transformer}'. "
            f"Re-run with trained_models.transformer={expected.split('/')[-1]}")

    # p_theta(x_i | x<i): the generative model from phase 1
    log.info("\n" + "-"*20 + "LANGUAGE MODEL (frozen)" + "-"*20 + "\n")
    lm, lm_cfg = load_trained_transformer(model_dir_transformer, vocab_size, device)
    log.info(lm)

    # p_phi(y | x<=i): the family head from phase 2
    log.info("\n" + "-"*20 + "CLASSIFIER HEAD" + "-"*20 + "\n")
    head, _ = load_trained_head(model_dir_head,
                                input_features=lm.embedding_dim,
                                output_features=n_classes,
                                device=device)
    log.info(head)

    # The two heads share the language model's hidden states
    classifier = ClassifierGuidedLM(lm, head).to(device)
    classifier.eval()
    log.info("\n" + "-"*20 + "GUIDED MODEL" + "-"*20 + "\n")
    log.info(classifier)

    # Parameters
    n_head = sum(p.numel() for p in head.parameters())
    n_lm   = sum(p.numel() for p in lm.parameters())
    log.info(f"Classifier Head: {n_head:,}   Language model (frozen): {n_lm:,}")

    # Let's try to generate some sequences
    log.info("\n" + "-"*20 + "GENERATION" + "-"*20 + "\n")

    max_seq_len = train_set.max_length
    bos_id = train_set.vocab["<BOS>"]
    eos_id = train_set.vocab["<EOS>"]
    pad_id = train_set.vocab["<PAD>"]

    n_seq           = cfg.generate.n_seq
    batch_size      = cfg.generate.batch_size
    temperature     = cfg.generate.temperature
    guidance_scale  = cfg.generate.guidance_scale

    # null family = no steering, the unguided baseline
    family = cfg.generate.family
    if family is None:
        target_idx = None
        guidance_scale = 0.0
        setting = "unguided"
        log.info("No family given: generating from the transformer alone (lambda = 0)")
    else:
        if family not in train_set.fam_to_idx:
            raise ValueError(f"Unknown family '{family}', expected one of {list(train_set.fam_to_idx)}")
        target_idx = train_set.fam_to_idx[family]
        setting = f"{family}_lambda{guidance_scale}"
        log.info(f"Steering towards '{family}' (class {target_idx}) with lambda = {guidance_scale}")


    log.info(f"Generating {n_seq} sequences of max length {max_seq_len}, batch size {batch_size}")
    log.info(f"Temperature: {temperature}")
    log.info(f"<BOS> = {bos_id}, <EOS> = {eos_id}, <PAD> = {pad_id}")

    sequences = torch.full((n_seq, 1), fill_value=bos_id, dtype=torch.long, device=device)
    
    # lm, head and classifier are already in eval mode from their loaders

    with torch.no_grad():

        for step in range(max_seq_len - 1):

            # log p_theta(x_i | x<i): hidden states -> the language model's own output head
            # (B, T, d)
            hidden = lm(sequences, return_hidden_states=True)
            # (B, V)
            # we apply the last output projection that gives us the prob of the next pos
            token_logits = lm.output_proj(hidden[:, -1, :])

            # <PAD> and <BOS> must never be sampled
            # So we put them to -infinity before the log softmax
            token_logits[:, [pad_id, bos_id]] = float("-inf")

            # (B, V)
            log_p = torch.log_softmax(token_logits / temperature, dim=-1)

            if target_idx is not None and guidance_scale != 0.0:

                B, T = sequences.shape
                V = vocab_size

                # Every candidate continuation: (B, T) -> (B*V, T+1)
                # (B*V, T)
                # Each row repeated V times back to back:[[1,5,7],[1,5,7],[1,5,7]] for as many candidate tokens
                # exist for the next position
                seq_rep  = sequences.repeat_interleave(V, dim=0)
                # (B*V,)
                # BAsically made as an index to identify which one is the candidate in the next prediction
                cand     = torch.arange(V, device=device).repeat(B)
                # (B*V, T+1)
                # Example:
                # [[1,5,7, 0],
                # [1,5,7, 1],
                # [1,5,7, 2],
                # [1,5,7, 3],
                # [1,9,3, 0],
                cand_seq = torch.cat([seq_rep, cand.unsqueeze(1)], dim=1)

                # log p_phi(y | x<=i): the family read from each candidate prefix
                # (B*V, C)

                # We get the logits for each one of the candidates
                fam_logits = classifier(cand_seq)[:, -1, :]
                
                # Use log softmax, catxau on the target_idx
                # TARGET ID in here, this is where we select the target that we are looking for 
                fam_logp   = torch.log_softmax(fam_logits, dim=-1)[:, target_idx]

                # (B, V)
                # In here we weight the steering, how much steering do we want is guided ny guidance_scale
                log_p = log_p + guidance_scale * fam_logp.view(B, V)

            # 
            probs = torch.softmax(log_p, dim=-1)
            residues = torch.multinomial(probs, 1)

            sequences = torch.cat((sequences, residues), 1)

            # Every sequence has produced <EOS> at least once
            if (sequences == eos_id).any(dim=1).all():
                log.info(f"All sequences finished at step {step + 1}")
                break

    # Suppressing all tokens after the first EOS 
    decoded_sequences = []

    for seq in sequences:
        # we're on a single sequence, so we need the first dimension of the tuple only
        eos_positions = (seq == eos_id).nonzero(as_tuple=True)[0] 
        
        if len(eos_positions) > 0:
            seq_cropped = seq[:eos_positions[0]] #stop after the first EOS token encountered
        #No EOS found after the max sequence length, we still conserve the sequence
        else:
            seq_cropped = seq 
        seq_decoded = train_set.decode(seq_cropped) 
        decoded_sequences.append(seq_decoded)


    log.info("\n" +"-"*20 + "DATA EXPORT" + "-"*20 + "\n")
    # The setting columns are what let every lambda run be concatenated into a single eval
    df_decoded = pd.DataFrame({
        "sequence": decoded_sequences,
        "setting":  setting,
        "family":   family if family is not None else "none",
        "lambda":   guidance_scale,
    })
    out_file = Path(out_dir) / f"sequences_{setting}.csv"
    df_decoded.to_csv(out_file, index=False)
    log.info(f"{len(decoded_sequences)} decoded sequences have been written to {out_file}!" + "\n")

    n_empty = sum(s == "" for s in decoded_sequences)
    if n_empty:
        log.info(f"warning: {n_empty} sequences decoded to an empty string (<EOS> at position 1)")



    # To take a look at how some sequences are predicted, before cropping and encoding them
    # sequences_to_plot = sequences[:10]
    # log.info(f"Example of generated sequences (still encoded):")
    # for i, seq in enumerate(sequences_to_plot):
    #     log.info(f"Seq {i}: {seq.tolist()}")

    # To take a look at how some sequences are predicted, AFTER cropping and encoding them
    # sequences_to_plot = decoded_sequences[:10]
    # log.info(f"Example of generated sequences (decoded):")
    # for i, seq in enumerate(sequences_to_plot):
    #     log.info(f"Seq {i}: {seq}")


        


if __name__ == "__main__":
    start = time.time()
    log.info(f"Started:  {datetime.now():%Y-%m-%d %H:%M:%S}")
    main()
    log.info(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"Duration: {(time.time() - start) / 60:.1f} min")