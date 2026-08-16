# Blood Test AI

A small neural network built from scratch with Python and NumPy to learn the fundamentals of machine learning.

The project uses synthetic blood test data to classify a patient into one of several categories. The goal is not to build a medically useful diagnostic system, but to understand how a neural network works internally.

## 🎯 Project Goals

This project was created as a learning exercise to understand:

- Neural network architecture
- Neurons and layers
- Weights and biases
- Matrix multiplication
- ReLU activation
- Logits
- Softmax
- Loss functions
- Backpropagation
- Gradient descent
- Training epochs
- Training vs. test datasets
- Model generalization and overfitting

The neural network is implemented manually using **NumPy**, without PyTorch or TensorFlow.

---

## 🧠 Neural Network Architecture

The current network has the following architecture:

```text
Input
  │
  │  5 features
  ▼
┌───────────────┐
│ Hidden Layer 1│
│  3 neurons    │
│    ReLU       │
└───────────────┘
  │
  │
  ▼
┌───────────────┐
│ Hidden Layer 2│
│  3 neurons    │
│    ReLU       │
└───────────────┘
  │
  │
  ▼
┌───────────────┐
│ Output Layer  │
│  5 neurons    │
└───────────────┘
  │
  ▼
 Logits
  │
  ▼
 Softmax
  │
  ▼
 Probabilities
 