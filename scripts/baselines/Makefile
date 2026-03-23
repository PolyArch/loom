# Build targets for GPU and CPU baseline benchmarks.
#
# Usage:
#   make cpu         - Build all CPU baselines
#   make gpu         - Build all GPU baselines (requires CUDA)
#   make all         - Build both
#   make run-cpu     - Build and run CPU baselines
#   make run-gpu     - Build and run GPU baselines
#   make run-all     - Build and run everything, then compare
#   make clean       - Remove binaries and results

# -----------------------------------------------------------------------
# Toolchain
# -----------------------------------------------------------------------

CXX      ?= g++
NVCC     ?= nvcc

CXXFLAGS  = -O3 -std=c++17 -fopenmp -march=native
NVCCFLAGS = -O3 -std=c++17 --use_fast_math

COMMON_DIR = common
INC_FLAGS  = -I.

# -----------------------------------------------------------------------
# CPU targets
# -----------------------------------------------------------------------

CPU_DIR     = cpu
CPU_SOURCES = matmul_cpu.cpp fft_cpu.cpp ntt_cpu.cpp attention_cpu.cpp spmv_cpu.cpp
CPU_BINS    = $(CPU_SOURCES:%.cpp=$(CPU_DIR)/%)

$(CPU_DIR)/%: $(CPU_DIR)/%.cpp $(COMMON_DIR)/timer.h
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) $< -o $@ -lm

.PHONY: cpu
cpu: $(CPU_BINS)
	@echo "CPU baselines built: $(CPU_BINS)"

# -----------------------------------------------------------------------
# GPU targets
# -----------------------------------------------------------------------

GPU_DIR     = gpu
GPU_SOURCES = matmul_gpu.cu fft_gpu.cu ntt_gpu.cu attention_gpu.cu spmv_gpu.cu

$(GPU_DIR)/matmul_gpu: $(GPU_DIR)/matmul_gpu.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcublas

$(GPU_DIR)/fft_gpu: $(GPU_DIR)/fft_gpu.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcufft

$(GPU_DIR)/ntt_gpu: $(GPU_DIR)/ntt_gpu.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(GPU_DIR)/attention_gpu: $(GPU_DIR)/attention_gpu.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcublas

$(GPU_DIR)/spmv_gpu: $(GPU_DIR)/spmv_gpu.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcusparse

GPU_BINS = $(GPU_DIR)/matmul_gpu $(GPU_DIR)/fft_gpu $(GPU_DIR)/ntt_gpu \
           $(GPU_DIR)/attention_gpu $(GPU_DIR)/spmv_gpu

.PHONY: gpu
gpu: $(GPU_BINS)
	@echo "GPU baselines built: $(GPU_BINS)"

# -----------------------------------------------------------------------
# Combined
# -----------------------------------------------------------------------

.PHONY: all
all: cpu gpu

# -----------------------------------------------------------------------
# Run targets
# -----------------------------------------------------------------------

.PHONY: run-cpu
run-cpu: cpu
	python3 $(CPU_DIR)/run_cpu_baselines.py

.PHONY: run-gpu
run-gpu: gpu
	python3 $(GPU_DIR)/run_gpu_baselines.py

.PHONY: run-all
run-all: run-cpu run-gpu
	python3 compare_all.py

.PHONY: compare
compare:
	python3 compare_all.py

# -----------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------

.PHONY: clean
clean:
	rm -f $(CPU_BINS) $(GPU_BINS)
	rm -rf results/
	@echo "Cleaned build artifacts and results."
