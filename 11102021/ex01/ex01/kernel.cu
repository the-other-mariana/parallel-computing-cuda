#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define N 3
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

__host__ void validate(int* result_CPU, int* result_GPU, int size) {
	if (*result_CPU != *result_GPU) {
		printf("The results are not equal.\n");
		return;
	}
	printf("Kernel validated successfully.\n");
	return;
}

__host__ void CPU_fn(int* v, int* sum, const int size) {
	for (int i = 0; i < size; i++) {
		*sum += v[i];
	}
}

__global__ void kernel(int* res) {
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
	int* host_A = (int*)malloc(sizeof(int) * N * N);
	cudaMalloc((void**)&dev_B, sizeof(int) * N * N);

	for (int i = 0; i < N * N; i++) {
		host_A[i] = i + 1;
	}

	cudaMemcpyToSymbol(dev_A, host_A, sizeof(int) * N * N);

	dim3 grid(1);
	dim3 block(N, N);
	kernel << < grid, block >> > (dev_B);
	cudaMemcpy(host_B, dev_B, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

	printMtx(host_A);
	cout << endl;
	printMtx(host_B);

	return 0;
}