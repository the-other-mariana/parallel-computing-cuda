#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__host__ void checkCUDAError(const char* msg) {
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), msg);
	}
}

__host__ void validate(int* result_CPU, int* result_GPU, int size) {
	if (*result_CPU != *result_GPU) {
		printf("The results are not equal.\n");
		return;
	}
	printf("Kernel validated successfully.\n");
	return;
}

__host__ void CPU_fn(int *v, int* sum, const int size) {
	for (int i = 0; i < size; i++) {
		*sum += v[i];
	}
}

__global__ void kernel(int* v, int* sum) {
	int gId = threadIdx.x;
	int step = blockDim.x;

	while (step) {
		if (gId < step) {
			v[gId] = v[gId] + v[gId + step];
		}
		step = step / 2;
	}
	if (gId == 0) {
		*sum = v[gId];
	}
}

int main() {

	const int size = 1024;
	int* v = (int*)malloc(sizeof(int) * size);
	int sumCPU = 0;
	int sumGPU = 0;

	int* dev_v, *sum;
	cudaMalloc((void**)&dev_v, sizeof(int) * size);
	cudaMalloc((void**)&sum, sizeof(int));

	for (int i = 0; i < size; i++) {
		v[i] = 1;
	}

	cudaMemcpy(dev_v, v, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(sum, &sumGPU, sizeof(int), cudaMemcpyHostToDevice);

	dim3 grid(1);
	dim3 block(size);
	
	kernel <<< grid, block >>> (dev_v, sum);
	cudaMemcpy(&sumGPU, sum, sizeof(int), cudaMemcpyDeviceToHost);
	printf("GPU sum: %d\n", sumGPU);

	CPU_fn(v, &sumCPU, size);
	printf("CPU sum: %d\n", sumCPU);

	validate(&sumCPU, &sumGPU, size);

	return 0;
}