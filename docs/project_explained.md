# Explained

Steering autoregressive generative models toward desired protein sequences.
 
## Background
 
### Data
 
We work with sequential data of the form:
 
```
x = (x_1, ..., x_L) = x_1:L
```
 
where `x_1:L` is a protein sequence of length `L` over the token alphabet `A`,
taken here to be the 20 canonical amino acids:
 
```
A C D E F G H I K L M N P Q R S T V W Y
```
 
so `|A| = 20`.
 
The state space `A^L` is the set of all possible length-`L` strings over this
alphabet. It has `20^L` elements:
 
| L   | number of sequences |
| --- | ------------------- |
| 2   | 400                 |
| 3   | 8,000               |
| 5   | 3.2 million         |
| 21  | 2.1 x 10^27         |
| 100 | 1.3 x 10^130        |
 
Notation for prefixes:
 
```
x_<i  =  (x_1, ..., x_{i-1})     the residues before position i
x_<1  =  ()                      empty: nothing precedes position 1
```
 
### Autoregressive model
 
An autoregressive model (ARM) over the discrete state space `A^L` factorizes
the joint distribution using the probability chain rule:
 
```
p(x_1:L) = product over i = 1..L of  p(x_i | x_<i)
```
 
This is an exact identity, not an approximation. No independence assumption is
made. Each factor `p(x_i | x_<i)` is a categorical distribution over the 20
tokens, produced by a softmax over the model's output logits.
 
### Worked example
 
Take the A chain of human insulin, `GIVEQCCTSICSLYQLENYCN` (L = 21). For the
first five residues:
 
```
p(GIVEQ) = p(G) * p(I|G) * p(V|GI) * p(E|GIV) * p(Q|GIVE)
```
 
Read each factor as "the probability of this residue, given everything before
it":
 
| factor     | meaning                                        |
| ---------- | ---------------------------------------------- |
| p(G)       | chance the sequence starts with G              |
| p(I \| G)    | chance the 2nd residue is I, given the 1st was G |
| p(V \| GI)   | chance the 3rd is V, given the first two were GI |
| p(E \| GIV)  | chance the 4th is E, given GIV came before     |
| p(Q \| GIVE) | chance the 5th is Q, given GIVE came before    |
 
Each factor is one entry pulled out of a 20-way softmax. At step 3, the model
is fed the prefix `GI` and outputs 20 probabilities summing to 1. The actual
third residue is `V`, so we read off that entry.
 
With illustrative numbers:
 
```
prefix seen   next   factor    running product
(start)       G      0.08      8.000e-02
G             I      0.06      4.800e-03
GI            V      0.11      5.280e-04
GIV           E      0.07      3.696e-05
GIVE          Q      0.09      3.326e-06
```
 
So `p(GIVEQ) = 3.3e-06`, against `20^-5 = 3.1e-07` for a uniform model. About
10x more probable than chance, over just five residues.
 
### Log space
 
Every factor is below 1, so the product shrinks fast. Over a 200-residue
protein it would underflow a float. In practice everything is computed in log
space, which turns the product into a sum:
 
```
log p(x_1:L) = sum over i = 1..L of  log p(x_i | x_<i)
```
 
The negative of this, averaged over positions, is the token-level
cross-entropy used as the training objective. Maximising the likelihood of
real sequences and minimising cross-entropy are the same instruction.
 
### Why the conditioning matters
 
Without it, the factorization would be:
 
```
p(x) = p(x_1) * p(x_2) * ... * p(x_L)         WRONG
```
 
This assumes residues are independent, so the model would know only background
amino-acid frequencies and nothing about which residues follow which. The
`| x_<i` is what lets it capture motifs, periodicity, and the structural
constraints that make a sequence foldable.

# Questions

- Why is it called auto regressive? Is it an automatic regression happening in here in any way?
- Our ARM is conditioned by the size of the input, do we use any type of padding? padded
- What are our ARM are they the prob inside the Multiplier or the full sequences probability?
- Which one is the desired conditional distribuiton?
- Confirm which one is the generation, which one the the predtictive and what is the guided, guided because we are conditioning on a certain label is that so?
