# ===============================
# Compiler settings (CPU / Windows legacy)
# ===============================
CC ?= cl
CFLAGS = /Idev /Zi /nologo /W4 /WX- /diagnostics:column /sdl /O2 /Oi /Ot /GL /D _DEBUG /D _CONSOLE /D _UNICODE /D UNICODE /Gm- /EHsc /MD /GS /Gy /fp:fast /Zc:wchar_t /Zc:forScope /Zc:inline /permissive- \
 /external:W3 /Gd /TP /wd4996 /Fd$@.pdb /FC /openmp:llvm
LDFLAGS =
LDLIBS =
INCLUDES =

# ===============================
# CUDA / NVCC settings
# ===============================
USE_CUDNN ?= 0
BUILD_DIR  = build

# ===============================
# Windows / Linux setup
# ===============================
ifeq ($(OS),Windows_NT)
  $(shell if not exist $(BUILD_DIR) mkdir $(BUILD_DIR))
  REMOVE_BUILD_OBJECT_FILES := del $(BUILD_DIR)\*.obj
  REMOVE_FILES    = del *.exe *.obj *.lib *.exp *.pdb
  OUTPUT_FILE     = /link /OUT:$@
  CUDA_OUTPUT_FILE = -o $@ && copy /Y $@.exe $@
  OBJ_EXT = obj
else
  $(shell mkdir -p $(BUILD_DIR))
  REMOVE_BUILD_OBJECT_FILES := rm -f $(BUILD_DIR)/*.o
  REMOVE_FILES    = rm -f
  OUTPUT_FILE     = -o $@
  CUDA_OUTPUT_FILE = -o $@
  OBJ_EXT = o
endif

# ===============================
# NVCC path (Windows / Linux)
# ===============================
ifeq ($(OS),Windows_NT)
  NVCC         := "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe"
  NVCC_FLAGS    = --threads=0 -t=0 --use_fast_math -std=c++17 -O3 -arch=sm_89
  NVCC_LDFLAGS  =
  NVCC_LDLIBS   = -lcublas -lcublasLt -lnvml
  NVCC_INCLUDES =
else
  # Pin to the CUDA 13.0 toolkit so nvcc, nvlink, libcudadevrt, cublas, etc.
  # all come from the same release — avoids "size does not match -m64" errors
  # when the system PATH nvcc is a different version.
  NVCC := /usr/local/cuda-13.0/bin/nvcc

  # -cudart=shared: dynamic CUDA runtime linkage, avoids static-archive ABI issues.
  NVCC_FLAGS    = --threads=0 -t=0 --use_fast_math -std=c++17 -O3 -arch=sm_89 -cudart=shared

  NVCC_INCLUDES = -I/usr/local/cuda-13.0/include \
                  -I/usr/include \
                  -I/usr/local/lib/python3.12/site-packages/nvidia/cublas/include \
                  -I/usr/local/lib/python3.12/site-packages/nvidia/cudart/include \
                  -I/usr/local/lib/python3.12/site-packages/nvidia/nvtx/include \
                  -I/usr/local/lib/python3.12/site-packages/nvidia/cudnn/include

  NVCC_LDFLAGS  = -L. -Xlinker -rpath -Xlinker \$$ORIGIN
  NVCC_LDFLAGS += -L/usr/local/cuda-13.0/lib64 -L/usr/local/lib \
                  -L/usr/lib/x86_64-linux-gnu \
                  -L/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib

  NVCC_LDLIBS   = -lcublas -lcublasLt -lnvml
endif

# ===============================
# cuDNN (optional, USE_CUDNN=1)
# ===============================
NVCC_CUDNN =
ifeq ($(USE_CUDNN),1)

ifeq ($(OS),Windows_NT)
  ifeq ($(shell if exist "$(HOMEDRIVE)$(HOMEPATH)\cudnn-frontend\include" (echo exists)),exists)
    CUDNN_FRONTEND_PATH = $(HOMEDRIVE)$(HOMEPATH)\cudnn-frontend\include
  else ifeq ($(shell if exist "cudnn-frontend\include" (echo exists)),exists)
    CUDNN_FRONTEND_PATH = cudnn-frontend/include
  else
    $(error [ERROR] cuDNN frontend not found. See README)
  endif
  CUDNN_INCLUDE_PATH = -I"C:\Program Files\NVIDIA\CUDNN\v9.17\include\13.1"
  CUDNN_LIB_PATH     = -L"C:\Program Files\NVIDIA\CUDNN\v9.17\lib\13.1\x64"
else
  ifeq ($(shell test -d $$HOME/cudnn-frontend/include && echo exists),exists)
    CUDNN_FRONTEND_PATH = $(HOME)/cudnn-frontend/include
  else ifeq ($(shell test -d cudnn-frontend/include && echo exists),exists)
    CUDNN_FRONTEND_PATH = cudnn-frontend/include
  else
    $(error [ERROR] cuDNN frontend not found. See README)
  endif
  ifneq ($(wildcard /usr/include/cudnn.h),)
    CUDNN_INCLUDE_PATH = -I/usr/include
  else ifneq ($(wildcard /usr/local/cuda-13.0/include/cudnn.h),)
    CUDNN_INCLUDE_PATH = -I/usr/local/cuda-13.0/include
  else ifneq ($(wildcard /usr/local/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn.h),)
    CUDNN_INCLUDE_PATH = -I/usr/local/lib/python3.12/site-packages/nvidia/cudnn/include
  else
    CUDNN_INCLUDE_PATH = -I/usr/local/include
  endif
  ifneq ($(wildcard /usr/lib/x86_64-linux-gnu/libcudnn.so),)
    CUDNN_LIB_PATH = -L/usr/lib/x86_64-linux-gnu
  else ifneq ($(wildcard /usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9),)
    CUDNN_LIB_PATH = -L/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib
  endif
endif

  NVCC_INCLUDES += -I$(CUDNN_FRONTEND_PATH) $(CUDNN_INCLUDE_PATH)
  NVCC_LDFLAGS  += $(CUDNN_LIB_PATH)
  NVCC_LDLIBS   += -lcudnn
  NVCC_FLAGS    += -DENABLE_CUDNN
  NVCC_CUDNN     = $(BUILD_DIR)/cudnn_att.$(OBJ_EXT)

else
  $(info → cuDNN disabled. Run make USE_CUDNN=1 to enable.)
endif

# ===============================
# SF16 / Q1.15 — always on
# Every model is trained in strict SF16 forward / BF16 backward mode.
# ===============================
SF16FLAGS = -DENABLE_BF16 -DENABLE_Q115 -DSF16_TRUE_FORWARD=1

# ===============================
# Phony targets
# ===============================
.PHONY: all clean libsyms \
        train_gpt2 train_gpt3 train_sfnet train_llama32_1B train_llama32_3B

all: train_gpt2 train_gpt3 train_sfnet train_llama32_1B train_llama32_3B

# ===============================
# Linux library symlinks (no-op on Windows)
# ===============================
ifneq ($(OS),Windows_NT)
libsyms:
	@if [ -f /usr/local/cuda-13.0/lib64/libcublas.so.12 ]; then \
		ln -sf /usr/local/cuda-13.0/lib64/libcublas.so.12    ./libcublas.so.12; \
		echo "Linked CUDA 13.0 cuBLAS"; \
	else \
		ln -sf /usr/local/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12 ./libcublas.so.12; \
		echo "Linked pip cuBLAS (fallback)"; \
	fi
	ln -sf ./libcublas.so.12 ./libcublas.so
	@if [ -f /usr/local/cuda-13.0/lib64/libcublasLt.so.12 ]; then \
		ln -sf /usr/local/cuda-13.0/lib64/libcublasLt.so.12  ./libcublasLt.so.12; \
	else \
		ln -sf /usr/local/lib/python3.12/site-packages/nvidia/cublas/lib/libcublasLt.so.12 ./libcublasLt.so.12; \
	fi
	ln -sf ./libcublasLt.so.12 ./libcublasLt.so
	ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 ./libnvml.so.1
	ln -sf ./libnvml.so.1 ./libnvml.so
else
libsyms: ;
endif

# ===============================
# cuDNN object (built only when USE_CUDNN=1)
# ===============================
$(NVCC_CUDNN): llmc/cudnn_att.cpp
	$(NVCC) -c $(NVCC_FLAGS) $(SF16FLAGS) $< $(NVCC_INCLUDES) -o $@

# ===============================
# Model targets — all SF16
# ===============================
train_gpt2: train_gpt2.cu libsyms $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(SF16FLAGS) $(NVCC_INCLUDES) $< $(NVCC_CUDNN) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_gpt3: train_gpt3.cu libsyms $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(SF16FLAGS) $(NVCC_INCLUDES) $< $(NVCC_CUDNN) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_sfnet: train_sfnet.cu libsyms
	$(NVCC) $(NVCC_FLAGS) $(SF16FLAGS) $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_llama32_1B: train_llama32_1B.cu libsyms
	$(NVCC) $(NVCC_FLAGS) $(SF16FLAGS) $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_llama32_3B: train_llama32_3B.cu libsyms
	$(NVCC) $(NVCC_FLAGS) $(SF16FLAGS) $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

# ===============================
# Clean
# ===============================
clean:
	$(REMOVE_FILES) train_gpt2 train_gpt3 train_sfnet train_llama32_1B train_llama32_3B \
	                libcublas.so libcublas.so.12 libcublasLt.so libcublasLt.so.12 \
	                libnvml.so libnvml.so.1 *.o
	$(REMOVE_BUILD_OBJECT_FILES)
