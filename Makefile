# ===============================
# Compiler settings
# ===============================
CC ?= cl
CFLAGS = /Idev /Zi /nologo /W4 /WX- /diagnostics:column /sdl /O2 /Oi /Ot /GL /D _DEBUG /D _CONSOLE /D _UNICODE /D UNICODE /Gm- /EHsc /MD /GS /Gy /fp:fast /Zc:wchar_t /Zc:forScope /Zc:inline /permissive- \
 /external:W3 /Gd /TP /wd4996 /Fd$@.pdb /FC /openmp:llvm
LDFLAGS =
LDLIBS =
INCLUDES =
CFLAGS_COND =

# ===============================
# CUDA / NVCC settings
# ===============================
FORCE_NVCC_O ?= 3
NVCC_CUDNN =

USE_CUDNN ?= 0
BUILD_DIR = build

# ===============================
# Windows / Linux setup
# ===============================
ifeq ($(OS),Windows_NT)
  $(shell if not exist $(BUILD_DIR) mkdir $(BUILD_DIR))
  REMOVE_BUILD_OBJECT_FILES := del $(BUILD_DIR)\*.obj
  REMOVE_FILES = del *.exe *.obj *.lib *.exp *.pdb
  OUTPUT_FILE = /link /OUT:$@
  CUDA_OUTPUT_FILE = -o $@ && copy /Y $@.exe $@
  OBJ_EXT = obj
else
  $(shell mkdir -p $(BUILD_DIR))
  REMOVE_BUILD_OBJECT_FILES := rm -f $(BUILD_DIR)/*.o
  REMOVE_FILES = rm -f
  OUTPUT_FILE = -o $@
  CUDA_OUTPUT_FILE = -o $@
  OBJ_EXT = o
endif

# ===============================
# NVCC path (Windows / Linux)
# ===============================
ifeq ($(OS),Windows_NT)
  NVCC ?= "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe"
  NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O$(FORCE_NVCC_O) -arch=sm_89
  NVCC_LDFLAGS =
  NVCC_LDLIBS  = -lcublas -lcublasLt -lnvml
  NVCC_INCLUDES =
else
  NVCC ?= $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
  NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O3 -arch=sm_89
  NVCC_INCLUDES = -I/usr/local/lib/python3.12/site-packages/nvidia/cublas/include \
                  -I/usr/local/lib/python3.12/site-packages/nvidia/cudart/include \
                  -I/usr/local/lib/python3.12/site-packages/nvidia/nvtx/include \
                  -I/usr/local/cuda/include
  # Link against local shim directory first.
  # Pass rpath via -Xlinker (NOT -Wl,...) so runtime can find libs next to the binary.
  NVCC_LDFLAGS = -L. -Xlinker -rpath -Xlinker \$$ORIGIN
  NVCC_LDLIBS  = -lcublas -lcublasLt -lnvml
endif

# ===============================
# cuDNN (Windows / Linux)
# ===============================
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

else  # ===== Linux =====

  ifeq ($(shell test -d $$HOME/cudnn-frontend/include && echo exists),exists)
    CUDNN_FRONTEND_PATH = $(HOME)/cudnn-frontend/include
  else ifeq ($(shell test -d cudnn-frontend/include && echo exists),exists)
    CUDNN_FRONTEND_PATH = cudnn-frontend/include
  else
    $(error [ERROR] cuDNN frontend not found. See README)
  endif

  CUDNN_INCLUDE_PATH = -I/usr/include
  CUDNN_LIB_PATH     = -L/usr/lib/x86_64-linux-gnu

endif

  NVCC_INCLUDES += -I$(CUDNN_FRONTEND_PATH) $(CUDNN_INCLUDE_PATH)
  NVCC_LDFLAGS  += $(CUDNN_LIB_PATH)
  NVCC_LDLIBS   += -lcudnn
  NVCC_FLAGS    += -DENABLE_CUDNN

  NVCC_CUDNN = $(BUILD_DIR)/cudnn_att.$(OBJ_EXT)

else
  $(info → cuDNN disabled by default. Run make USE_CUDNN=1 to enable)
endif

# ===============================
# Precision settings
# ===============================
PRECISION ?= BF16
VALID_PRECISIONS := FP32 FP16 BF16
ifeq ($(filter $(PRECISION),$(VALID_PRECISIONS)),)
  $(error Invalid precision $(PRECISION), valid precisions are $(VALID_PRECISIONS))
endif

ifeq ($(PRECISION),FP32)
  PFLAGS = -DENABLE_FP32
else ifeq ($(PRECISION),FP16)
  PFLAGS = -DENABLE_FP16
else
  PFLAGS = -DENABLE_BF16
endif

# ===============================
# Targets
# ===============================
TARGETS = train_gpt2 test_gpt2 train_gpt2cu train_gpt2rawcu train_gpt3cu test_gpt2cu train_gpt2fp32cu test_gpt2fp32cu $(NVCC_CUDNN)

TARGETS_Q131 = train_gpt2q131cu train_gpt3q131cu
TARGETS_Q115 = train_gpt2q115cu train_gpt3q115cu
TARGETS_Q115_CONSTRAINED = train_gpt2q115_constrainedcu train_gpt3q115_constrainedcu

.PHONY: all clean libsyms run q131 q115 q115_constrained
all: $(TARGETS)
q131: $(TARGETS_Q131)
q115: $(TARGETS_Q115)
q115_constrained: $(TARGETS_Q115_CONSTRAINED)

# ===============================
# Linux library symlinks
# (no-op on Windows)
# ===============================
ifneq ($(OS),Windows_NT)
libsyms:
	ln -sf /usr/local/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12    ./libcublas.so.12
	ln -sf ./libcublas.so.12                                                            ./libcublas.so
	ln -sf /usr/local/lib/python3.12/site-packages/nvidia/cublas/lib/libcublasLt.so.12  ./libcublasLt.so.12
	ln -sf ./libcublasLt.so.12                                                          ./libcublasLt.so
	ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1                                   ./libnvml.so.1
	ln -sf ./libnvml.so.1                                                               ./libnvml.so
else
libsyms: ;
endif

# ===============================
# CPU targets
# ===============================
train_gpt2: train_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

test_gpt2: test_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

# ===============================
# CUDA targets
# ===============================
$(NVCC_CUDNN): llmc/cudnn_att.cpp
	$(NVCC) -c $(NVCC_FLAGS) $(PFLAGS) $< $(NVCC_INCLUDES) -o $@

train_gpt2cu: train_gpt2.cu libsyms $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $< $(NVCC_CUDNN) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_gpt2rawcu: train_gpt2_raw.cu libsyms $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $< $(NVCC_CUDNN) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_gpt3cu: train_gpt3.cu libsyms $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $< $(NVCC_CUDNN) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_gpt2fp32cu: train_gpt2_fp32.cu libsyms
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

test_gpt2cu: test_gpt2.cu libsyms $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $< $(NVCC_CUDNN) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

test_gpt2fp32cu: test_gpt2_fp32.cu libsyms
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

profile_gpt2cu: profile_gpt2.cu libsyms $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -lineinfo $(NVCC_INCLUDES) $< $(NVCC_CUDNN) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

# ===============================
# Quantized CUDA targets (Q1.31)
# ===============================
train_gpt2q131cu: train_gpt2.cu libsyms $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -DENABLE_Q131 -DFIXED_POINT_Q31 $(NVCC_INCLUDES) $< $(NVCC_CUDNN) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_gpt3q131cu: train_gpt3.cu libsyms $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -DENABLE_Q131 -DFIXED_POINT_Q31 $(NVCC_INCLUDES) $< $(NVCC_CUDNN) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

# ===============================
# Quantized CUDA targets (Q1.15)
# ===============================
train_gpt2q115cu: train_gpt2.cu libsyms $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -DENABLE_Q115 $(NVCC_INCLUDES) $< $(NVCC_CUDNN) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_gpt3q115cu: train_gpt3.cu libsyms $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -DENABLE_Q115 $(NVCC_INCLUDES) $< $(NVCC_CUDNN) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

# ===============================
# Q1.15 Weight-Constrained CUDA
# ===============================
train_gpt2q115_constrainedcu: train_gpt2.cu libsyms $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -DENABLE_Q115_WEIGHT_CONSTRAINT $(NVCC_INCLUDES) $< $(NVCC_CUDNN) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_gpt3q115_constrainedcu: train_gpt3.cu libsyms $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -DENABLE_Q115_WEIGHT_CONSTRAINT $(NVCC_INCLUDES) $< $(NVCC_CUDNN) $(NVCC_LDFLAGS) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

# ===============================
# Convenience run (Linux only)
# ===============================
run: train_gpt2cu
	./train_gpt2cu --help

# ===============================
# Clean
# ===============================
clean:
	$(REMOVE_FILES) train_gpt2cu train_gpt2fp32cu train_gpt2q131cu train_gpt2q115cu *.o \
	      libcublas.so libcublas.so.12 libcublasLt.so libcublasLt.so.12 libnvml.so libnvml.so.1
	$(REMOVE_BUILD_OBJECT_FILES)