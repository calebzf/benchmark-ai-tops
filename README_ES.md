# Herramienta de Benchmark de Rendimiento de IA

Una suite de benchmark de nivel profesional diseñada para validar las capacidades completas de IA de **Apple Silicon (M1/M2/M3/M4)**. Unifica pruebas de CPU, GPU (Metal) y NPU (Neural Engine) en una sola herramienta.

Este script implementa estrategias de ingeniería avanzadas como **Inferencia Residente en Caché** y **Cuantización de Pesos INT8** para medir el verdadero potencial del Apple Neural Engine (ANE).

## 🚀 Características Principales

* **Análisis de Espectro Completo:**
    * **CPU (NumPy):** Mide el rendimiento bruto de punto flotante (GFLOPS).
    * **GPU (Metal MPS):** Prueba el caudal de Cómputo (FP32) e Inferencia (FP16).
    * **NPU (CoreML):** Utiliza `coremltools` para evitar cuellos de botella de Python y acceder directamente al Neural Engine.
* **Estrategias NPU Avanzadas:**
    * **Modelo Residente en Caché:** Usa capas profundas con tensores pequeños (32x32) para evitar cuellos de botella en RAM y saturar la SRAM interna del NPU.
    * **Cuantización INT8:** Aplica cuantización lineal de pesos para desbloquear la lógica de aceleración del Neural Engine.
* **Detección de Hardware:** Detecta automáticamente núcleos Físicos/Lógicos, cantidad de núcleos de GPU y estado del driver NPU.

## Requisitos Previos

* **Python 3.10 o 3.11** (Requerido para la compatibilidad de TensorFlow/PyTorch en macOS ARM64).
* **Arquitectura:** ARM64 (Apple Silicon) o x86_64.

## Instalación y Configuración

1.  **Crear un entorno virtual limpio:**
    ```bash
    # Verifique que está usando Python 3.10 o 3.11
    python3.10 -m venv venv

    # Activar el entorno
    source venv/bin/activate
    ```

2.  **Instalar dependencias:**
    *Nota: Requerimos estrictamente `numpy<2` para evitar conflictos con TensorFlow.*
    ```bash
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    ```

## Uso

Ejecute el script directamente desde su terminal:

```bash
python benchmark-ai.py
```


## Entendiendo los Resultados
CPU Baseline (GFLOPS): Rendimiento estándar de punto flotante en el procesador.

FP32 (TFLOPS): Potencia bruta de cómputo GPU. Alta precisión, usada para entrenamiento o cálculo científico.

FP16 (TOPS): Potencia de Inferencia de IA. Menor precisión, mayor velocidad. Esta métrica se alinea más con las especificaciones de marketing de NPU/Neural Engine.

## 📝 Resultado de Ejemplo (Apple M4 Pro)

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
🏆  INFORME TÉCNICO DE RENDIMIENTO (M4 PRO)
================================================================================
• CPU (Procesamiento General):   340.12 GFLOPS
• GPU (Gráficos / IA Básica):    7.83 TOPS
• NPU (IA Alta Precisión):       14.34 TOPS
• NPU (IA Cuantizada W8A16):     18.23 TOPS
--------------------------------------------------------------------------------
NOTA: El resultado de ~18.23 TOPS representa el ~50% del pico teórico.
Esta es la velocidad máxima posible sin un dataset de calibración (modo W8A16).
Para alcanzar el máximo de TOPS completos (W8A8), se requiere un modelo real entrenado
con cuantización de activaciones.
================================================================================
```