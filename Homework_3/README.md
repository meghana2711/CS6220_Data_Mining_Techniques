# CS 6220: Homework 3 — Machine Learning & Data Mining

> **This notebook implements and evaluates k-Nearest Neighbors (k-NN) with general $L_p$ distance metrics, Rosenblatt's Perceptron, and a generalized single-unit neuron from scratch, alongside a gradient flow analysis of deep neural networks on MNIST to explore the vanishing gradient problem and bias-variance tradeoff.**

## 1. Environment Setup & Prerequisites

This project requires **Python 3.10+** (tested on Python 3.14). All required libraries can be installed via `pip`:

```bash
pip install -r requirements.txt
```

### Key Dependencies
- `numpy`: Tensor mathematics and distance computation
- `scipy`: Numerical processing
- `matplotlib`: Plotting decision boundaries, error curves, and gradient flow
- `scikit-learn`: `make_moons` dataset generator and baseline benchmarks
- `torch` & `torchvision`: Neural network implementation and MNIST gradient flow analysis
- `ipykernel` & `notebook`: Jupyter Notebook runtime

---

## 3. How to Run the Notebook

To execute the notebook interactively:

1. Launch Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook CS6220_Homework_3.ipynb
   ```
2. In the top navigation bar, select **Kernel -> Restart & Run All**.
3. All cells will sequentially execute:
   - **Part 1:** k-NN classification, decision boundary visualization, and LOOCV error curves.
   - **Part 2:** Rosenblatt Perceptron training and generalized single-unit neuron continuous regression.
   - **Part 3:** Deep MLP gradient flow simulation comparing Sigmoid vs. ReLU and MSE vs. Cross-Entropy loss.

---