#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define N 32
__constant__ int dev_A[N * N];

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
	for (int i = 0; i < N * N; i++) {
		if (*result_CPU != *result_GPU) {
			printf("[FAILED] The results are not equal.\n");
			return;
		}
	}
	printf("[SUCCESS] Kernel validation.\n");
	return;
}

__host__ void CPU_transpose(int* vector, int* res) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res[(i * N) + j] = vector[(N * j) + i];
		}
	}
}

__global__ void GPU_transpose(int* res) {
	int gId = threadIdx.x + (blockDim.x * threadIdx.y);
	res[gId] = dev_A[N * threadIdx.x + threadIdx.y];
}

__host__ void printMtx(int* mtx) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << mtx[(i * N) + j] << " ";
		}
		cout << endl;
	}
}

int main() {

	int* dev_B;
	int* host_B = (int*)malloc(sizeof(int) * N * N);
	int* cpu_B = (int*)malloc(sizeof(int) * N * N);
	int* host_A = (int*)malloc(sizeof(int) * N * N);

	cudaMalloc((void**)&dev_B, sizeof(int) * N * N);
	checkCUDAError("Error at cudaMalloc: dev_B");

	for (int i = 0; i < N * N; i++) {
		host_A[i] = i + 1;
	}

	cudaMemcpyToSymbol(dev_A, host_A, sizeof(int) * N * N);
	checkCUDAError("Error at MemcpyToSymbol");

	dim3 grid(1);
	dim3 block(N, N);
	GPU_transpose << < grid, block >> > (dev_B);
	checkCUDAError("Error at kernel");
	cudaMemcpy(host_B, dev_B, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
	checkCUDAError("Error at Memcpy host_B <- dev_B");

	CPU_transpose(host_A, cpu_B);

	printf("Input: \n");
	printMtx(host_A);
	printf("CPU: \n");
	printMtx(cpu_B);
	printf("GPU: \n");
	printMtx(host_B);

	validate(cpu_B, host_B);

	free(host_B);
	free(cpu_B);
	free(host_A);
	cudaFree(dev_B);

	return 0;
}