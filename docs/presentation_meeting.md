# Presentation Meeting

### Objectives for today:

- getting up and running
- get to use dtu hpc to run our models (relevant)
- they talked about a bit of 101 biology

### Goal of our projects:

- create a model to generate some sequences
- steer the models so that they can have some properties (steer here means applying pressure, conditioned them to generate something in certain directions)

### Groups of proteins we will be working with:

- phosphatase, phosphocarrier, DNA-binding (We will want to steer to each one of the groups)

### Data:

- They will give use the data already split to ensure that there is no data leakage

### Autoregressive Modeling:

- Next token prediction (similar to large language models)
- Per position probabilities
- Optimize the model to do maximum likelihood 
- We will use various models: MLP (baseline), RNN, Transformer (we can decide if we want to implement the RNN or the Transformer)
- Then the model generates protein sequences (for the 3 different families)
- Some guided autoregression -> guidance to generate proteins from a certain family

### Classification

- Classifier Guidance (log stuff I am a bit lost)
- Classifeir free guidance (You are not giving ir any label)

### Final Validation

- Use ESMFOld's, measure of hte foadability of this sequences. Model own, how confident are we, that this is a real protein

## Practicals

- DTU HPC, log in and all
- Clone repo in the root and let'ss do do do lo do do do
- In the powerpoint presentation we have a suggested 
- We need to look at the git, we have so much stuff there!!!!!!!!!!!!!!!!!!!!!!!!!


## Questions

- Most likely, steer for this protein family (do we do this for each step)
- generative, needs to be able to predict under the subparts (not the full lenght)


- use only 3 residues to predict the next one - slide across sequences to predict the next one

- RNN is this a LSTM or just RNN
- Transformer, do we need to chose 1 between both?
- look better to tokenization of proteins, special tokens and paddings, same number and all, 0, start 1, end 2 (look into the git repo)
- test if the simlar residues  are close in the space. Take the embedings, visualize, saw some clustering. - Identifiable, chemical 
- check the paper mlp



