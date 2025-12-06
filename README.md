# Custom CUDA Transformer Inference Engine ⚡

![C++](https://img.shields.io/badge/Language-C%2B%2B-blue)
![CUDA](https://img.shields.io/badge/Hardware-NVIDIA_T4-green)
![PyTorch](https://img.shields.io/badge/Verification-PyTorch_Match-orange)

A bare-metal, zero-dependency C++ inference runtime for Transformer blocks (BERT/GPT architecture). 

This project bypasses high-level frameworks (PyTorch) to interact directly with the GPU via **CUDA Runtime API**, manually managing memory hierarchy and executing custom compute kernels.

## 📸 System Dashboard
*Real-time verification of C++ execution vs PyTorch reference.*

![Dashboard](https://github.com/Adityajl/cuda-transformer-engine/blob/master/assets/dashboard_view.png)


## 🔧 Architecture
The engine implements a standard Feed-Forward Network (FFN) pipeline using custom CUDA kernels:
1.  **Layer Normalization:** Parallel reduction algorithm.
2.  **Linear Projection:** Tiled Matrix Multiplication with shared memory.
3.  **GELU Activation:** `tanh` approximation matching GPT-2 spec.

## 📊 Numerical Verification
We validated the CUDA output against a PyTorch reference implementation ("Golden Data").

![Verification](assets/verification.png)

### The "Non-Associativity" Problem
You will notice a max error of `~5.8e-05`. This is **expected behavior** in High-Performance Computing.
*   **CPU (PyTorch):** Sums numbers sequentially.
*   **GPU (CUDA):** Sums numbers in parallel blocks.
*   **Result:** Due to floating-point non-associativity, the order of operations alters the rounding bits slightly. 

## 💻 How to Run
The entire source code, compilation steps, and visualization pipeline are contained in the Jupyter Notebook.

1.  Open `CUDA_Transformer_Inference.ipynb` in Google Colab.
2.  Set Runtime to **T4 GPU**.
3.  Run All Cells.

---
*Author: Aditya Jaiswal*
