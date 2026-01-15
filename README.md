# Online-softmax-fused-kernel-

This repository contains a high-performance CUDA kernel that implements a **Fused Online Softmax and Matrix Multiplication**. By combining these two operations, the kernel minimizes global memory access and improves numerical stability, similar to the optimizations found in **FlashAttention**.

---

## 🚀 Overview

In standard deep learning pipelines, Softmax and Matrix Multiplication (MatMul) are separate kernels. This requires writing the MatMul results to Global Memory (HBM) and reading them back for Softmax. 

This implementation uses the **Online Softmax** algorithm, which allows us to normalize the data as we compute the dot product, keeping the "running state" entirely within GPU registers and shared memory.

### Key Benefits
* **Memory Efficiency:** Reduces HBM traffic by fusing the normalization step.
* **Numerical Stability:** Uses a running maximum to prevent `exp(x)` from overflowing.
* **Work Coarsening:** Processes multiple output elements per thread (`COARSE_FACTOR = 16`) to maximize instruction-level parallelism.

---

## 🧠 The Algorithm: Online Softmax



To avoid multiple passes over the data, we track three variables per row:
1.  **Running Max ($m$):** The largest value seen so far.
2.  **Running Sum ($d$):** The sum of exponentials, rescaled to the current max.
3.  **Accumulator ($acc$):** The partial MatMul result, rescaled to the current max.

As we encounter a new value $x$, we update the state:

$$m_{new} = \max(m_{old}, x)$$
$$d_{new} = d_{old} \cdot e^{m_{old} - m_{new}} + e^{x - m_{new}}$$
$$acc_{new} = acc_{old} \cdot e^{m_{old} - m_{new}} + e^{x - m_{new}} \cdot V_{ij}$$

---

## 🛠 Technical Specifications



### Configuration Constants
| Constant | Value | Description |
| :--- | :--- | :--- |
| `TILE_WIDTH` | 32 | Dimension of square tiles in shared memory. |
| `COARSE_FACTOR` | 16 | Number of horizontal output elements per thread. |

### Kernel Signature
```cpp
__global__ void OnlineSotmax_matmul(float* A, float* V, float* Out, int width)
