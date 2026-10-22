# SYMMET - Symmetry-Aware Machine Learning

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
