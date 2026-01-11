# AI Performance Benchmark Tool

A professional-grade benchmarking suite designed to validate the full AI capabilities of **Apple Silicon (M1/M2/M3/M4)**. It unifies CPU, GPU (Metal), and NPU (Neural Engine) testing into a single tool.

This script implements advanced engineering strategies like **Cache-Resident Inference** and **INT8 Weight Quantization** to measure the true potential of the Apple Neural Engine (ANE).

## 🚀 Key Features

* **Full Spectrum Analysis:**
    * **CPU (NumPy):** Measures raw floating-point performance (GFLOPS).
    * **GPU (Metal MPS):** Tests Compute (FP32) and Inference (FP16) throughput.
    * **NPU (CoreML):** Utilizes `coremltools` to bypass Python bottlenecks and access the Neural Engine directly.
* **Advanced NPU Strategies:**
    * **Cache-Resident Model:** Uses deep layers with small tensors (32x32) to prevent RAM bottlenecks and saturate the NPU's internal SRAM.
    * **INT8 Quantization:** Applies linear weight quantization to unlock the Neural Engine's acceleration logic.
* **Hardware Detection:** Auto-detects Physical/Logical cores, GPU Core count, and NPU driver status.

## Prerequisites

* **Python 3.10 or 3.11** (Required for TensorFlow/PyTorch compatibility on macOS ARM64).
* **Architecture:** ARM64 (Apple Silicon) or x86_64.

## Installation & Setup

1.  **Create a clean virtual environment:**
    ```bash
    # Verify you are using Python 3.10 or 3.11
    python3.10 -m venv venv

    # Activate the environment
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    *Note: We strictly require `numpy<2` to prevent conflicts with TensorFlow.*
    ```bash
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    ```

## Usage

Run the script directly from your terminal:

```bash
python benchmark-ai.py
```

## Understanding the Output
CPU Baseline (GFLOPS): Standard floating-point performance on the processor.

FP32 (TFLOPS): Raw GPU Compute power. High precision, used for training or scientific calc.

FP16 (TOPS): AI Inference power. Lower precision, faster speed. This metric aligns closer with NPU/Neural Engine marketing specs.


## 📝 Example Result (Apple M4 Pro)

```bash
================================================================================
🚀  AI BENCHMARK PRO
================================================================================
OS: Darwin 24.6.0 | RAM: 24.0 GB
CPU: arm (12 Physical / 12 Logical)
GPU: MPS (16 Cores) | NPU: Enabled
--------------------------------------------------------------------------------
[1] CPU BASELINE (FP32)... 340.12 GFLOPS
[2] GPU METAL (FP16)...... 7.83 TOPS
[3] NPU NEURAL (FP16)..... 14.34 TOPS
[4] NPU NEURAL (INT8)..... 18.23 TOPS

================================================================================
🏆  INFORME TÉCNICO DE RENDIMIENTO
================================================================================
• CPU (General Processing):      340.12 GFLOPS
• GPU (Graphics / Basic AI):     7.83 TOPS
• NPU (High Precision AI):       14.34 TOPS
• NPU (Quantized AI W8A16):      18.23 TOPS
--------------------------------------------------------------------------------
NOTE: The ~18.23 TOPS result represents ~50% of the theoretical peak.
This is the maximum speed achievable without a calibration dataset (W8A16 mode).
To reach full TOPS (W8A8), a real-world trained model with activation
quantization is required.
================================================================================
```

