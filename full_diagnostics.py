"""
========================================================
PyTorch Full Diagnostics + Performance Benchmarks
Author  : Arpit Sachin Kadam
Purpose : System validation, GPU benchmarking, attention
          profiling, and FP16 vs BF16 comparison for
          Transformer / LLM workloads
========================================================
"""

import torch
import time
import platform

# =======================================================
# 1. SYSTEM INFORMATION
# =======================================================
print("=" * 90)
print("PYTORCH FULL DIAGNOSTICS & BENCHMARK SUITE")
print("=" * 90)

print("\n[1] SYSTEM INFORMATION")
print(f"Operating System           : {platform.system()} {platform.release()}")
print(f"Python Version             : {platform.python_version()}")

# =======================================================
# 2. PYTORCH BUILD INFORMATION
# =======================================================
print("\n[2] PYTORCH BUILD INFORMATION")
print(f"PyTorch Version            : {torch.__version__}")
print(f"PyTorch Git Version        : {torch.version.git_version}")
print(f"Compiled CUDA Version      : {torch.version.cuda}")

# =======================================================
# 3. CUDA AVAILABILITY
# =======================================================
print("\n[3] CUDA AVAILABILITY")
print(f"CUDA Available             : {torch.cuda.is_available()}")
print(f"CUDA Device Count          : {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available — GPU benchmarks require CUDA")

device = "cuda"

# =======================================================
# 4. GPU DEVICE DETAILS
# =======================================================
print("\n[4] GPU DEVICE DETAILS")

device_id = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device_id)

print(f"GPU Name                   : {props.name}")
print(f"Compute Capability         : {props.major}.{props.minor}")
print(f"Total GPU Memory (GB)      : {props.total_memory / 1024**3:.2f}")

# =======================================================
# 5. CUDA DRIVER & RUNTIME
# =======================================================
print("\n[5] CUDA DRIVER & RUNTIME")
print(f"CUDA Runtime Version       : {torch._C._cuda_getCompiledVersion()}")

# =======================================================
# 6. GPU MEMORY STATUS
# =======================================================
print("\n[6] GPU MEMORY STATUS")
print(f"Memory Allocated (MB)      : {torch.cuda.memory_allocated() / 1024**2:.2f}")
print(f"Memory Reserved (MB)       : {torch.cuda.memory_reserved() / 1024**2:.2f}")
print(f"Peak Memory Used (MB)      : {torch.cuda.max_memory_allocated() / 1024**2:.2f}")

# =======================================================
# 7. PRECISION & ACCELERATION FEATURES
# =======================================================
print("\n[7] PRECISION & ACCELERATION")
print(f"BF16 Supported             : {torch.cuda.is_bf16_supported()}")
print(f"TF32 Enabled (MatMul)      : {torch.backends.cuda.matmul.allow_tf32}")
print(f"Flash Attention Enabled    : {torch.backends.cuda.flash_sdp_enabled()}")
print(f"Mem-Efficient Attention    : {torch.backends.cuda.mem_efficient_sdp_enabled()}")

# =======================================================
# 8. CPU BACKEND OPTIMIZATIONS
# =======================================================
print("\n[8] CPU BACKEND OPTIMIZATIONS")
print(f"CPU Threads Used           : {torch.get_num_threads()}")
print(f"OpenMP Available           : {torch.backends.openmp.is_available()}")
print(f"MKL Available              : {torch.backends.mkl.is_available()}")
print(f"MKLDNN Available           : {torch.backends.mkldnn.is_available()}")

# =======================================================
# 9. DISTRIBUTED TRAINING SUPPORT
# =======================================================
print("\n[9] DISTRIBUTED TRAINING")
print(f"Distributed Available      : {torch.distributed.is_available()}")
print(f"NCCL Backend Available     : {torch.distributed.is_nccl_available()}")

# =======================================================
# 10. QUANTIZATION BACKENDS
# =======================================================
print("\n[10] QUANTIZATION BACKENDS")
print(f"Active Quantization Engine : {torch.backends.quantized.engine}")
print(f"Supported Engines          : {torch.backends.quantized.supported_engines}")

# =======================================================
# HELPER FUNCTION: SYNCHRONIZED TIMER
# =======================================================
def timed(fn, iters=10):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) / iters

# =======================================================
# 11. MATMUL TFLOPS BENCHMARK
# =======================================================
print("\n[11] MATMUL TFLOPS BENCHMARK")

N = 4096
dtype = torch.float16

a = torch.randn(N, N, device=device, dtype=dtype)
b = torch.randn(N, N, device=device, dtype=dtype)

def matmul():
    torch.matmul(a, b)

t = timed(matmul, iters=20)
flops = 2 * (N ** 3)
tflops = flops / t / 1e12

print(f"Matrix Size                : {N} x {N}")
print(f"Data Type                  : FP16")
print(f"Avg Time per MatMul (s)    : {t:.6f}")
print(f"Achieved TFLOPs            : {tflops:.2f}")

# =======================================================
# 12. TOKENS / SECOND BENCHMARK
# =======================================================
print("\n[12] TOKENS / SECOND BENCHMARK")

batch_size = 8
seq_len = 2048
hidden_dim = 4096

x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
linear = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(device).half()

def transformer_step():
    return linear(x)

t = timed(transformer_step, iters=20)
tokens = batch_size * seq_len
tokens_per_sec = tokens / t

print(f"Batch Size                 : {batch_size}")
print(f"Sequence Length            : {seq_len}")
print(f"Hidden Dimension           : {hidden_dim}")
print(f"Throughput (tokens/sec)    : {tokens_per_sec:,.0f}")

# =======================================================
# 13. ATTENTION SPEED COMPARISON
# =======================================================
print("\n[13] ATTENTION SPEED COMPARISON")

batch = 4
heads = 16
seq = 2048
head_dim = 64

q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
k = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
v = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)

def attention():
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

results = {}

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
results["Standard (Math)"] = timed(attention, iters=10)

if torch.backends.cuda.mem_efficient_sdp_enabled():
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    results["Memory-Efficient"] = timed(attention, iters=10)

if torch.backends.cuda.flash_sdp_enabled():
    torch.backends.cuda.enable_flash_sdp(True)
    results["Flash Attention"] = timed(attention, iters=10)

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

for k, v in results.items():
    print(f"{k:<25} : {v:.6f} sec")

# =======================================================
# 14. BF16 vs FP16 COMPARISON
# =======================================================
print("\n[14] BF16 vs FP16 PERFORMANCE COMPARISON")

if not torch.cuda.is_bf16_supported():
    print("BF16 not supported — skipping comparison.")
else:
    dtypes = {"FP16": torch.float16, "BF16": torch.bfloat16}
    results = {}

    print("\n[14.1] MATMUL TFLOPS")
    for name, dt in dtypes.items():
        a = torch.randn(N, N, device=device, dtype=dt)
        b = torch.randn(N, N, device=device, dtype=dt)
        t = timed(lambda: torch.matmul(a, b), iters=20)
        results[(name, "matmul")] = 2 * N**3 / t / 1e12
        print(f"{name:<6} | TFLOPs: {results[(name,'matmul')]:.2f}")

    print("\n[14.2] TOKENS / SECOND")
    for name, dt in dtypes.items():
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dt)
        linear = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(device, dt)
        t = timed(lambda: linear(x), iters=20)
        results[(name, "tokens")] = (batch_size * seq_len) / t
        print(f"{name:<6} | Tokens/sec: {results[(name,'tokens')]:,.0f}")

    print("\n[14.3] ATTENTION LATENCY")
    for name, dt in dtypes.items():
        q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dt)
        k = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dt)
        v = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dt)
        t = timed(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True), iters=10)
        results[(name, "attention")] = t
        print(f"{name:<6} | Avg Time: {t:.6f}s")

# =======================================================
# 15. REPRODUCIBILITY
# =======================================================
print("\n[15] REPRODUCIBILITY")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
print("Random Seed Set            : 42")

# =======================================================
# 16. Power / clock reporting (nvidia-smi integration)
# =======================================================
print("\n[16] POWER / CLOCK REPORTING (NVIDIA-SMI)")

import subprocess

def run_nvidia_smi(query):
    """
    Runs an nvidia-smi query and returns parsed output.
    Falls back gracefully if nvidia-smi is unavailable.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout.strip().split(", ")
    except Exception as e:
        print("nvidia-smi not available or failed to execute.")
        return None

# -------------------------------------------------------
# Query definitions
# -------------------------------------------------------
queries = {
    "Power Draw (W)"        : "power.draw",
    "Power Limit (W)"      : "power.limit",
    "GPU Temperature (C)"  : "temperature.gpu",
    "SM Clock (MHz)"       : "clocks.sm",
    "Memory Clock (MHz)"   : "clocks.mem",
    "GPU Utilization (%)"  : "utilization.gpu",
    "Memory Util (%)"      : "utilization.memory"
}

# -------------------------------------------------------
# Execute queries
# -------------------------------------------------------
results = {}
for label, q in queries.items():
    val = run_nvidia_smi(q)
    if val is not None:
        results[label] = val[0]

# -------------------------------------------------------
# Print results
# -------------------------------------------------------
if results:
    for k, v in results.items():
        print(f"{k:<25} : {v}")
else:
    print("Power / clock metrics unavailable.")

# Usage:
# - Correlates performance with power draw
# - Helps detect throttling (power or thermal)
# - Essential for laptop GPUs and long training runs

print("\n" + "=" * 90)
print("ALL DIAGNOSTICS & BENCHMARKS COMPLETED SUCCESSFULLY")
print("=" * 90)
