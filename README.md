# SYMMET - Symmetry-Aware Machine Learning

## Reserch Question
Can geometric knowledge improve neural network sample efficiency? 

## Hypothesis
Model with geometric iductive bias will lean circular classification boundaries using a smaller dataset, causing the training to be both cheeper and faster.

## Models
- A: Baseline MLP -> 337 Parameters
- B: Radial MLP -> 321 Parameters
- C: Radial Logistic Regression -> 2 Parameters

## Running Experiments
`python main.py`