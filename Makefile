# ============================================================================
# Makefile — Pipeline CHASM em C + CUDA
# UFG — Computação de Alto Desempenho
#
# Uso:
#   make all          → compila tudo (gerador, CPU, pipeline GPU, benchmark)
#   make kernels      → compila apenas os .so CUDA
#   make data         → gera as imagens sintéticas
#   make benchmark    → roda o benchmark completo
#   make clean        → remove binários e dados
# ============================================================================

CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -std=c11
LDFLAGS = -lm -ldl
NVCC    = nvcc
NVCCFLAGS = -O3 -arch=sm_89 -Xcompiler -fPIC

# ── Binários C ───────────────────────────────────────────────────────────────
all: glcm_data glcm_cpu glcm_pipeline benchmark kernels

glcm_data: glcm_data.c
	$(CC) $(CFLAGS) -o $@ $< -lm

glcm_cpu: glcm_cpu.c glcm_common.h
	$(CC) $(CFLAGS) -o $@ $< -lm

glcm_pipeline: glcm_pipeline.c glcm_common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

benchmark: benchmark.c glcm_cpu.c glcm_pipeline.c glcm_common.h
	$(CC) $(CFLAGS) -o $@ benchmark.c glcm_cpu.c glcm_pipeline.c $(LDFLAGS)

# ── Kernels CUDA (.so) ────────────────────────────────────────────────────────
kernels: glcm_cuda.so glcm_shared.so glcm_shared_v2.so glcm_dynpar.so

glcm_cuda.so: glcm_cuda.cu
	$(NVCC) $(NVCCFLAGS) -shared -o $@ $<

glcm_shared.so: glcm_shared.cu
	$(NVCC) $(NVCCFLAGS) -shared -o $@ $<

glcm_shared_v2.so: glcm_shared_v2.cu
	$(NVCC) $(NVCCFLAGS) -shared -o $@ $<

glcm_dynpar.so: glcm_dynpar.cu
	$(NVCC) $(NVCCFLAGS) -rdc=true -shared -o $@ $< \
		-lcudadevrt -lcudart_static -lrt -lpthread -ldl

# ── Targets de conveniência ───────────────────────────────────────────────────
data: glcm_data
	./glcm_data

run_benchmark: benchmark data
	./benchmark

clean:
	rm -f glcm_data glcm_cpu glcm_pipeline benchmark
	rm -f *.so benchmark_results.json
	rm -rf data/

.PHONY: all kernels data run_benchmark clean
