
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

__global__ void matrixSum(int* dev_a, int* dev_b, int* dev_c) {
	int gId = threadIdx.x + threadIdx.y * blockDim.x;
	dev_c[gId] = dev_a[gId] + dev_b[gId];
}

int main() {
	const int N = 3; // if 32 ok, if 33 ERROR 9: invalid configuration argument (matrixSum kernel error)

	int* host_a = (int*)malloc(sizeof(int) * N * N);
	int* host_b = (int*)malloc(sizeof(int) * N * N);
	int* host_c = (int*)malloc(sizeof(int) * N * N);

	int* dev_a, * dev_b, * dev_c;
	cudaMalloc((void**)&dev_a, sizeof(int) * N * N);
	cudaMalloc((void**)&dev_b, sizeof(int) * N * N);
	cudaMalloc((void**)&dev_c, sizeof(int) * N * N);

	// init data
	for (int i = 0; i < N * N; i++) {
		host_a[i] = (int)(rand() % 10);
		host_b[i] = (int)(rand() % 10);
	}

	cudaMemcpy(dev_a, host_a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, sizeof(int) * N * N, cudaMemcpyHostToDevice);

	dim3 block(N, N);
	dim3 grid(1);

	matrixSum << < grid, block >> > (dev_a, dev_b, dev_c);
	checkCUDAError("matrixSum kernel error");

	cudaMemcpy(host_c, dev_c, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

	printf("\nMatrix A: \n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d ", host_a[j + i * N]);
		}
		printf("\n");
	}

	printf("\nMatrix B: \n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d ", host_b[j + i * N]);
		}
		printf("\n");
	}

	printf("\nMatrix C: \n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d ", host_c[j + i * N]);
		}
		printf("\n");
	}

	free(host_a);
	free(host_b);
	free(host_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

