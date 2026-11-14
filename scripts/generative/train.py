import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

@hydra.main(config_path="./../../config", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print("-"*20 + "CONFIGURATION" + "-"*20 + "\n")
    print(OmegaConf.to_yaml(cfg))

    import torch
    from torch.utils.data import DataLoader
    from protgen.data.make_data import ProteinDataset
    from protgen.utils import set_seed
    
    # Setting the seed for the run
    # Default is 42 in the ~/config/default.yaml
    set_seed(cfg.seed)

    # Device to use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device being used for the run: {device} \n")

    # Data loading
    # Not loading testing in here to keep testing and training separate
    print("-"*23 + "DATA" + "-"*23 + "\n")
    train_set = ProteinDataset(cfg.data.train)
    print(train_set)
    val_set = ProteinDataset(cfg.data.val)
    print(val_set)
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle= True)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle= True)

    # Create the model
    print("\n" +"-"*23 + "MODEL" + "-"*23 + "\n")
    vocab_size = len(train_set.vocab)
    print(f"Vocabulary size: {vocab_size}")
    model = instantiate(cfg.model, vocab_size=vocab_size).to(device)
    print(model)
    print("-" * 48 + "\n")

    # Create the Optimizer
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    

if __name__ == "__main__":
    main()