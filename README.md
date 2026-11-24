# SYMMET - Symmetry-Aware Machine Learning
Disclamer: This paper was developed independently without reference to any prior research and makes no claim of originality; its findings are alredy established as core concepts in machine learning.

## Research Paper Available
At: [SYMMET - Symmetry-Aware Machine Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7650698)
## Research Question
Can geometric knowledge improve neural network sample efficiency? 

## Hypothesis
Model with geometric inductive bias will learn circular classification boundaries using a smaller dataset, causing the training to be both cheaper and faster.

## Models
- A: Baseline MLP -> 337 Parameters
- B: Radial MLP -> 321 Parameters
- C: Radial Logistic Regression -> 2 Parameters

## Running Experiments
`python main.py`
