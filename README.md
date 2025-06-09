# manual-nn-regression

# Neural Network from Scratch (NumPy + PyTorch)
A simple yet complete neural network for regression, built entirely from scratch using NumPy and PyTorch **without autograd**. Covers loss functions, mini-batch training, backpropagation, and manual gradient calculation.

## Why I Built This

I wanted to truly understand how neural networks learn — not just use libraries, but **implement the logic myself**.

So I built a 1-hidden-layer neural network using:
- Manual weight updates (no autograd)
- Custom implementations of MSE, MAE, and Huber loss
- Matrix-based forward and backward passes
- PyTorch tensors for faster math

This helped me understand every step of training — especially backpropagation, ReLU gradients, and how loss functions affect learning.

## How It Works

The model is a simple feedforward neural network with:
- 2 input features
- 1 hidden layer with 3 neurons (ReLU activation)
- 1 output neuron (for regression)

### Training Flow:
1. Forward pass (matrix multiplication + ReLU)
2. Compute loss (MSE)
3. Manually compute gradients using chain rule
4. Update weights with gradient descent
5. Repeat for 100 epochs

Loss reduces steadily, and the model learns to predict target values accurately.

## How to Run

Clone this repo and run the files in order:

1. `1_loss_functions/*.py` — MSE, MAE, Huber graphs  
2. `2_linear_regression/*.py` — Gradient descent on 1D linear model  
3. `3_nn_numpy/*.py` — Neural network in NumPy with manual backprop  
4. `4_nn_pytorch_manual/*.py` — PyTorch version using tensors only

Or open in **Google Colab** using this link: [https://colab.research.google.com/drive/1jGN6F-GDojPjzqCZ993cr1lk5jzbypE4?usp=sharing]

## Results

Here's the training loss over time:

![loss curve](https://github.com/sheshashai/manual-nn-regression/blob/main/loss_plot.png)

Final MSE: **3.25**  
Sample predictions:  

## What's Next

Next, I’ll:
- Add autograd support using PyTorch
- Train on a real-world dataset
- Create a simple UI using Gradio
- Start publishing more of my learning on LinkedIn and GitHub

