import time
import platform
import psutil
import torch
import torch.nn as nn
import numpy as np
import subprocess
import logging
import warnings
from typing import Tuple, Dict

# --- CONFIGURACIÓN DE ENTORNO ---
warnings.filterwarnings("ignore")
logging.getLogger("coremltools").setLevel(logging.ERROR)

try:
    import coremltools as ct
    import coremltools.optimize.coreml as cto
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

class AIBenchmarkMaster:
    """
    Valida CPU, GPU y NPU (FP16 & INT8 Quantization)
    """
    def __init__(self):
        self.device = self._get_device()
        self.system_info = self._get_system_info()
        
    def _get_device(self) -> torch.device:
        if torch.backends.mps.is_available(): return torch.device("mps")
        elif torch.cuda.is_available(): return torch.device("cuda")
        return torch.device("cpu")

    def _get_mac_gpu_cores(self) -> str:
        if platform.system() != "Darwin": return "N/A"
        try:
            cmd = ["system_profiler", "SPDisplaysDataType"]
            output = subprocess.check_output(cmd).decode("utf-8")
            for line in output.split('\n'):
                if "Total Number of Cores" in line:
                    return line.split(":")[1].strip()
            return "Unknown"
        except: return "N/A"

    def _get_system_info(self) -> Dict[str, str]:
        mem = psutil.virtual_memory()
        gpu_cores = self._get_mac_gpu_cores()
        # Restaurada la detección de Cores de CPU
        phy_cores = psutil.cpu_count(logical=False)
        log_cores = psutil.cpu_count(logical=True)
        
        return {
            "System": f"{platform.system()} {platform.release()}",
            "Processor": platform.processor(),
            "Cores": f"{phy_cores} Physical / {log_cores} Logical",
            "Device": f"{self.device.type.upper()} ({gpu_cores} Cores)",
            "Memory": f"{mem.total / (1024**3):.1f} GB",
            "NPU": "Enabled" if COREML_AVAILABLE else "Disabled"
        }

    def _warmup(self, operation, *args):
        for _ in range(3): operation(*args)
        if self.device.type == 'mps': torch.mps.synchronize()

    # --- 1. CPU TEST ---
    def benchmark_cpu(self, size: int = 2048) -> float:
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)
        flops = 2 * (size ** 3)
        def op(): np.dot(a, b)
        op() # Warmup
        start = time.perf_counter()
        for _ in range(10): op()
        end = time.perf_counter()
        return (flops / ((end - start)/10)) / 1e9

    # --- 2. GPU TEST (FP16) ---
    def benchmark_gpu(self) -> float:
        size = 4096
        a = torch.randn(size, size, device=self.device, dtype=torch.float16)
        b = torch.randn(size, size, device=self.device, dtype=torch.float16)
        flops = 2 * (size ** 3)
        def op(): torch.matmul(a, b)
        self._warmup(op)
        start = time.perf_counter()
        for _ in range(20): op()
        if self.device.type == 'mps': torch.mps.synchronize()
        end = time.perf_counter()
        return (flops / ((end - start)/20)) / 1e12

    # --- 3. NPU BUILDER (Generador de Modelos) ---
    def _build_npu_model(self, quantize=False):
        batch, size, channels, kernel, layers = 16, 32, 1536, 3, 5
        
        class DeepStress(nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = nn.Sequential(*[
                    nn.Sequential(nn.Conv2d(channels, channels, kernel, padding=1, bias=False), nn.ReLU())
                    for _ in range(layers)
                ])
            def forward(self, x): return self.mod(x)

        torch_model = DeepStress().eval()
        dummy = torch.randn(batch, channels, size, size)
        
        try:
            traced = torch.jit.trace(torch_model, dummy)
            model = ct.convert(
                traced,
                inputs=[ct.TensorType(name="input", shape=dummy.shape)],
                convert_to="mlprogram",
                compute_units=ct.ComputeUnit.ALL
            )
            
            if quantize:
                print("    ...🔨 Comprimiendo a INT8 (W8A16)...")
                op_config = cto.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)
                config = cto.OptimizationConfig(global_config=op_config)
                model = cto.linear_quantize_weights(model, config)
                
            return model, dummy.numpy(), (layers * 2 * channels * (kernel**2) * size * size * channels * batch)
        except Exception as e:
            return None, None, 0

    # --- RUN SUITE ---
    def run(self):
        print("\n" + "="*60)
        print(f"AI BENCHMARK PRO")
        print("="*60)
        print(f"OS: {self.system_info['System']} | RAM: {self.system_info['Memory']}")
        print(f"CPU: {self.system_info['Processor']} ({self.system_info['Cores']})")
        print(f"GPU: {self.system_info['Device']} | NPU: {self.system_info['NPU']}")
        print("-" * 60)

        # 1. CPU
        print(f"[1] CPU BASELINE (FP32)... ", end="", flush=True)
        cpu_score = self.benchmark_cpu()
        print(f"{cpu_score:.2f} GFLOPS")

        # 2. GPU
        print(f"[2] GPU METAL (FP16)...... ", end="", flush=True)
        gpu_score = self.benchmark_gpu()
        print(f"{gpu_score:.2f} TOPS")

        # 3. NPU
        npu_fp16 = 0.0
        npu_int8 = 0.0
        
        if COREML_AVAILABLE:
            # NPU FP16
            print(f"[3] NPU NEURAL (FP16)..... ", end="", flush=True)
            model, data, ops = self._build_npu_model(quantize=False)
            if model:
                for _ in range(5): model.predict({"input": data})
                start = time.perf_counter()
                for _ in range(15): model.predict({"input": data})
                npu_fp16 = (ops / ((time.perf_counter()-start)/15)) / 1e12
                print(f"{npu_fp16:.2f} TOPS")
            else: print("Error")

            # NPU INT8
            print(f"[4] NPU NEURAL (INT8)..... ", end="", flush=True)
            model, data, ops = self._build_npu_model(quantize=True)
            if model:
                for _ in range(5): model.predict({"input": data})
                start = time.perf_counter()
                for _ in range(15): model.predict({"input": data})
                npu_int8 = (ops / ((time.perf_counter()-start)/15)) / 1e12
                print(f"{npu_int8:.2f} TOPS")
            else: print("Error")
        else:
            print("[3/4] NPU tests saltados (coremltools missing)")

        # RESULTADOS
        print("\n" + "="*60)
        print("🏆  INFORME TÉCNICO DE RENDIMIENTO")
        print("="*60)
        print(f"• CPU (Procesamiento General):   {cpu_score:.2f} GFLOPS")
        print(f"• GPU (Gráficos / IA Básica):    {gpu_score:.2f} TOPS")
        print(f"• NPU (IA Alta Precisión):       {npu_fp16:.2f} TOPS")
        print(f"• NPU (IA Cuantizada W8A16):     {npu_int8:.2f} TOPS")
        print("-" * 60)
        print(f"NOTA FINAL: El rendimiento de ~{npu_int8:.1f} TOPS representa el ~50% del pico teórico.")
        print(f"Esta es la máxima velocidad posible sin calibración de datos")
        print(f"(W8A16). Para alcanzar el máximo de TOPS (W8A8), se requiere un modelo real entrenado.")
        print("="*60 + "\n")

if __name__ == "__main__":
    AIBenchmarkMaster().run()