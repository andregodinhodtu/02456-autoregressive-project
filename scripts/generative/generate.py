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
    
    train_cfg = OmegaConf.load(model_dir / ".hydra" / "config.yaml")

    model = instantiate(train_cfg.model, vocab_size) 
    
    state = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, train_cfg

@hydra.main(config_path="./../../config/generate", config_name="default", version_base=None)
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
    test_set = ProteinDataset(cfg.data.test)
    log.info(test_set)
    test_loader = DataLoader(test_set, batch_size=cfg.test.batch_size, shuffle= False)
    vocab_size = len(test_set.vocab)
    log.info(f"Vocabulary size: {vocab_size}")

    # Import the model (depending on the config parameters used)
    log.info("\n" +"-"*23 + "MODEL" + "-"*23 + "\n")
    model, train_cfg = load_trained_model(model_dir, vocab_size=vocab_size) 
    log.info(model)


    # Let's try to generate some sequences
    log.info("\n" +"-"*20 + "GENERATION" + "-"*20 + "\n")

    x, _, _, _ = next(iter(test_loader))
    max_seq_len = x.shape[1]

    n_seq = cfg.generate.batch_size
    log.info(f"Generating {n_seq} sequences of max length {max_seq_len}")
    bos_id = test_set.vocab["<BOS>"]
    log.info(f"Token used for beginning of sequence: {bos_id}")
    sequences = torch.full((n_seq, 1), fill_value=bos_id, dtype=torch.long, device=device)
    
    model.eval()

    with torch.no_grad():
        for pred_round in range(max_seq_len -1): 
            sequences = sequences.to(device)
            outputs = model(sequences) 
            # maybe we could also do outputs = model(sequences[:, -model.context, :])
            # So that we will spare ressources (just providing the windonw for the next residues)

            last_pos = outputs[:, -1, :]
            probs = torch.nn.functional.softmax(last_pos, 1)
            residues = torch.multinomial(probs, 1)

            sequences = torch.cat((sequences, residues), 1)

        sequences_to_plot = sequences[:10]
        log.info(f"Example of generated seqeunces (still encoded): {sequences_to_plot}")


        


if __name__ == "__main__":
    start = time.time()
    log.info(f"Started:  {datetime.now():%Y-%m-%d %H:%M:%S}")
    main()
    log.info(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"Duration: {(time.time() - start) / 60:.1f} min")