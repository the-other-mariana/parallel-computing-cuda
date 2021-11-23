#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define numBlocks 8
#define threadsPerBlock 1024

using namespace std;

__host__ void checkCUDAError(const char* msg) {
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), msg);
	}
}

__host__ void validate(int* result_CPU, int* result_GPU) {
	if (*result_CPU != *result_GPU) {
		printf("[FAILED] Kernel validation.\n");
		return;
	}
	printf("[SUCCESS] Kernel validation.\n");
	return;
}

__host__ void CPU_reduction(int* v, int* sum) {
	for (int i = 0; i < numBlocks * threadsPerBlock; i++) {
		*sum += v[i];
	}
}

__global__ void GPU_reduction(int* v, int* sum) {
	__shared__ int vector[numBlocks * threadsPerBlock];
	int gId = threadIdx.x + blockDim.x * blockIdx.x;

	vector[gId] = v[gId];
	__syncthreads();
	int step = threadsPerBlock / 2;
	while (step) {
		if (threadIdx.x < step) {
			vector[gId] = vector[gId] + vector[gId + step];
			__syncthreads();
		}
		step = step / 2;
	}
	__syncthreads();
	if (threadIdx.x == 0) { // copy the partial results to the global mem vector
		//printf("SM->vector[%d]: %d\n", gId, vector[gId]);
		v[gId] = vector[gId];
		__syncthreads();
		//printf("GM->v[%d]: %d\n", gId, v[gId]);
	}
	if (gId < numBlocks) { // choose the first 4 threads to copy in the first 4 cells the partial sums
		v[gId] = v[gId * threadsPerBlock]; // 0 <- 0*8, 1 <- 1*8 ...
		__syncthreads();
		//printf("%d<-%d\n", v[gId], v[gId * threadsPerBlock]);
	}
	int new_step = numBlocks / 2;
	while (new_step) {
		if (gId < new_step) {
			v[gId] += v[gId + new_step];
		}
		new_step = new_step / 2;
	}
	__syncthreads();
	if (gId == 0) {
		*sum = v[gId];
	}
}

int main() {

	int* dev_a, * dev_sum;
	int host_sum = 0, CPU_sum = 0;
	int* host_a = (int*)malloc(sizeof(int) * numBlocks * threadsPerBlock);
	cudaMalloc((void**)&dev_a, sizeof(int) * numBlocks * threadsPerBlock);
	cudaMalloc((void**)&dev_sum, sizeof(int));

	for (int i = 0; i < numBlocks * threadsPerBlock; i++) {
		host_a[i] = 1;
	}

	cudaMemcpy(dev_a, host_a, sizeof(int) * numBlocks * threadsPerBlock, cudaMemcpyHostToDevice);

	dim3 grid(numBlocks, 1, 1);
	dim3 block(threadsPerBlock, 1, 1);
	GPU_reduction << < grid, block >> > (dev_a, dev_sum);
	cudaDeviceSynchronize();
	checkCUDAError("Error at kernel");

	printf("N: %d\n", numBlocks * threadsPerBlock);
	cudaMemcpy(&host_sum, dev_sum, sizeof(int), cudaMemcpyDeviceToHost);
	printf("GPU result: %d\n", host_sum);

	CPU_reduction(host_a, &CPU_sum);
	printf("CPU result: %d\n", CPU_sum);

	validate(&CPU_sum, &host_sum);

	return 0;
}