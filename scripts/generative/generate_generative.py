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
import pandas as pd
import matplotlib.pyplot as plt
import json
import os  

from torch.utils.data import DataLoader
from protgen.data.make_data import ProteinDataset, IGNORE_INDEX
from protgen.utils import set_seed

log = logging.getLogger(__name__)

def load_trained_model(model_dir: Path, vocab_size, device: str = "cpu"):
    
    train_cfg = OmegaConf.load(model_dir / ".hydra" / "config.yaml")

    model = instantiate(train_cfg.model, vocab_size) 
    
    state = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, train_cfg

@hydra.main(config_path="./../../config/generate_generative", config_name="default", version_base=None)
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

    model_dir = Path(f"results/models/{cfg.trained_model.name}")
    if not model_dir.exists():
        raise FileNotFoundError(f"No trained model at {model_dir}")

    # Data loading
    # Here only to get dynamic variables about the dataset: vocab_size, sequence length, id of the beginning of sequence token
    log.info("-"*23 + "DATA" + "-"*23 + "\n")
    train_set = ProteinDataset(cfg.data.train)
    log.info(train_set)
    train_loader = DataLoader(train_set, batch_size=cfg.generate.batch_size, shuffle= False)
    vocab_size = len(train_set.vocab)
    log.info(f"Vocabulary size: {vocab_size}")

    # Import the model (depending on the config parameters used)
    log.info("\n" +"-"*23 + "MODEL" + "-"*23 + "\n")
    model, train_cfg = load_trained_model(model_dir, vocab_size=vocab_size) 
    log.info(model)


    # Let's try to generate some sequences
    log.info("\n" +"-"*20 + "GENERATION" + "-"*20 + "\n")

    x, _, _, _ = next(iter(train_loader))
    max_seq_len = x.shape[1]

    n_seq = cfg.generate.batch_size
    log.info(f"Generating {n_seq} sequences of max length {max_seq_len}")
    bos_id = train_set.vocab["<BOS>"]
    eos_id = train_set.vocab["<EOS>"]
    log.info(f"Token used for beginning of sequence: {bos_id}")
    log.info(f"Token used for end of sequence: {eos_id}")
    sequences = torch.full((n_seq, 1), fill_value=bos_id, dtype=torch.long, device=device)
    
    model.eval()

    with torch.no_grad():
        for pred_round in range(max_seq_len -1): #idk if the EOS and BOS token should be included in the sequence length, the range can be adjusted
            sequences = sequences.to(device)
            outputs = model(sequences) 
            # maybe we could also do outputs = model(sequences[:, -model.context, :])
            # So that we will spare ressources (just providing the windonw for the next residues)

            last_pos = outputs[:, -1, :]
            probs = torch.nn.functional.softmax(last_pos, 1)
            residues = torch.multinomial(probs, 1)

            sequences = torch.cat((sequences, residues), 1)

    # Suppressing all tokens after the first EOS 
    decoded_sequences = []

    for seq in sequences:
        eos_positions = (seq == eos_id).nonzero(as_tuple=True)[0] # we're on a single sequence, so we need the first dimension of the tuple only
        
        if len(eos_positions) > 0:
            seq_cropped = seq[:eos_positions[0]] #stop after the first EOS token encountered
        else: #No EOS found after the max sequence length, we still conserve the sequence
            seq_cropped = seq 
        seq_decoded = train_set.decode(seq_cropped) 
        decoded_sequences.append(seq_decoded)


    log.info("\n" +"-"*20 + "DATA EXPORT" + "-"*20 + "\n")
    df_decoded = pd.DataFrame({"sequence": decoded_sequences})
    out_file = Path(f"results/sequences/{cfg.trained_model.name}_{n_seq}seq.csv") #we could redirect to another folder to keep the important outputs in the gitignore
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_decoded.to_csv(out_file, index=False)
    log.info(f"{n_seq} decoded sequences have been written to {out_file}!"+ "\n")



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