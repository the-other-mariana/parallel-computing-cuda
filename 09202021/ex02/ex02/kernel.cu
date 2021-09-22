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

__global__ void kernel(int* m, int* r) {
	int gId = threadIdx.x + threadIdx.y * blockDim.x;
	int n1 = gId - 1;
	int n2 = gId + 1;
	int n3 = gId - blockDim.x;
	int n4 = gId + blockDim.x;
	//printf("%d: %d %d %d %d\n", gId, n1, n2, n3, n4);
	if (n1 < 0 || n3 < 0 || n2 >= (blockDim.x * blockDim.x) || n4 >= (blockDim.x * blockDim.x) || gId % blockDim.x == 0 || (gId + 1) % blockDim.x == 0) {
		return;
	}
	else {
		int avg = (m[n1] + m[n2] + m[n3] + m[n4]) / 4;
		//printf("%d: %d %d %d %d\n", gId, m[n1], m[n2], m[n3], m[n4]);

		r[gId] = avg;
	}
}

int main() {
	const int size = 5;

	int m[size][size] = { 0 };
	int r[size][size] = { 0 };
	int m_vec[size * size] = { 0 };
	int r_vec[size * size] = { 0 };

	int* dev_m, * dev_r;
	cudaMalloc((void**)&dev_m, sizeof(int) * size * size);
	cudaMalloc((void**)&dev_r, sizeof(int) * size * size);

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			m[i][j] = (int)(rand() % 10);
			m_vec[j + i * size] = m[i][j];
		}
	}

	printf("Original Matrix:\n");
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("%d ", m[i][j]);
		}
		printf("\n");
	}

	for (int i = 0; i < size * size; i++) {
		//printf("%d ", m_vec[i]);
	}
	printf("\n");

	cudaMemcpy(dev_m, m_vec, sizeof(int) * size * size, cudaMemcpyHostToDevice);

	dim3 grid(1);
	dim3 block(size, size);
	kernel << < grid, block >> > (dev_m, dev_r);
	checkCUDAError("Error at kernel");

	cudaMemcpy(r_vec, dev_r, sizeof(int) * size * size, cudaMemcpyDeviceToHost);

	printf("Average Matrix:\n");
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			r[i][j] = r_vec[j + i * size];
			printf("%d ", r[i][j]);
		}
		printf("\n");
	}
}