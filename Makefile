# ===============================
# CUDA / NVCC settings (Modal prebuilt image)
# ===============================
NVCC ?= $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O3 -Wno-deprecated-gpu-targets

# Headers (you already validated these exist)
NVCC_INCLUDES = -I/usr/local/lib/python3.12/site-packages/nvidia/cublas/include \
                -I/usr/local/lib/python3.12/site-packages/nvidia/cudart/include \
                -I/usr/local/lib/python3.12/site-packages/nvidia/nvtx/include \
                -I/usr/local/cuda/include

# Link against local shim directory first.
# Pass rpath via -Xlinker (NOT -Wl,...) so runtime can find libs next to the binary. [web:235]
NVCC_LDFLAGS = -L. -Xlinker -rpath -Xlinker \$$ORIGIN
NVCC_LDLIBS  = -lcublas -lcublasLt -lnvml

PRECISION ?= BF16
ifeq ($(PRECISION),FP32)
  PFLAGS = -DENABLE_FP32
else ifeq ($(PRECISION),FP16)
  PFLAGS = -DENABLE_FP16
else
  PFLAGS = -DENABLE_BF16
endif

.PHONY: all clean libsyms run q131 q115
all: train_gpt2cu

# Q1.31 fixed-point build (32-bit weights, higher precision)
q131: train_gpt2q131cu

# Q1.15 fixed-point build (16-bit weights, legacy)
q115: train_gpt2q115cu

# Create local linker + runtime symlinks (no changes to system dirs)
libsyms:
	ln -sf /usr/local/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12    ./libcublas.so.12
	ln -sf ./libcublas.so.12                                                            ./libcublas.so
	ln -sf /usr/local/lib/python3.12/site-packages/nvidia/cublas/lib/libcublasLt.so.12  ./libcublasLt.so.12
	ln -sf ./libcublasLt.so.12                                                          ./libcublasLt.so
	ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1                                   ./libnvml.so.1
	ln -sf ./libnvml.so.1                                                               ./libnvml.so

train_gpt2cu: train_gpt2.cu libsyms
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

train_gpt2fp32cu: train_gpt2_fp32.cu libsyms
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

# Q1.31 fixed-point build (32-bit, higher precision than Q1.15)
train_gpt2q131cu: train_gpt2.cu libsyms
	$(NVCC) $(NVCC_FLAGS) -DENABLE_Q131 -DFIXED_POINT_Q31 $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

# Q1.15 fixed-point build (16-bit, legacy)
train_gpt2q115cu: train_gpt2.cu libsyms
	$(NVCC) $(NVCC_FLAGS) -DENABLE_Q115 $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

# Convenience run (should work even when invoked manually as ./train_gpt2cu)
run: train_gpt2cu
	./train_gpt2cu --help

clean:
	rm -f train_gpt2cu train_gpt2fp32cu train_gpt2q131cu train_gpt2q115cu *.o \
	      libcublas.so libcublas.so.12 libcublasLt.so libcublasLt.so.12 libnvml.so libnvml.so.1