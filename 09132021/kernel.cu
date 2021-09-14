#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h> /* srand, rand */
#include <time.h> /* time */

#include<iostream>
using namespace std;

__host__ void checkCUDAError(const char* msg) {
	cudaError_t error;
	cudaDeviceSynchronize(); 
	error = cudaGetLastError(); 
	if (error != cudaSuccess) {
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), msg);
	}
}

__global__ void idKernel(int* vecA, int* vecB, int* vecC) {
	int gId = threadIdx.x + blockDim.x * blockIdx.x;

	vecA[gId] = threadIdx.x;
	vecB[gId] = blockIdx.x;
	vecC[gId] = gId;
}

void printArray(int* arr, int size, char* msg) {
	cout << msg << ": ";
	for (int i = 0; i < size; i++) {
		printf("%d ", arr[i]);
	}
	printf("\n");
}

int main()
{
	const int vectorSize = 64;
	int* host_a = (int*)malloc(sizeof(int) * vectorSize);
	int* host_b = (int*)malloc(sizeof(int) * vectorSize);
	int* host_c = (int*)malloc(sizeof(int) * vectorSize);

	int* dev_a, * dev_b, * dev_c;

	cudaMalloc((void**)&dev_a, sizeof(int) * vectorSize);
	checkCUDAError("Error at cudaMalloc for dev_a");
	cudaMalloc((void**)&dev_b, sizeof(int) * vectorSize);
	checkCUDAError("Error at cudaMalloc for dev_b");
	cudaMalloc((void**)&dev_c, sizeof(int) * vectorSize);
	checkCUDAError("Error at cudaMalloc for dev_c");

	srand(time(NULL));

	for (int i = 0; i < vectorSize; i++) {
		host_a[i] = 0;
		host_b[i] = 0;
		host_c[i] = 0;
	}

	cudaMemcpy(dev_a, host_a, sizeof(int) * vectorSize, cudaMemcpyHostToDevice);
	checkCUDAError("Error at cudaMemcpy for host_a to dev_a");
	cudaMemcpy(dev_b, host_b, sizeof(int) * vectorSize, cudaMemcpyHostToDevice);
	checkCUDAError("Error at cudaMemcpy for host_b to dev_b");
	cudaMemcpy(host_c, dev_c, sizeof(int) * vectorSize, cudaMemcpyHostToDevice); // error 1
	checkCUDAError("Error at cudaMemcpy for host_c to dev_c");

	dim3 grid(1, 1, 1);
	dim3 block(2000, 1, 1); // max num is 1024, so here we will force an error
	idKernel << < grid, block >> > (dev_a, dev_b, dev_c);
	checkCUDAError("Error at idKernel execution no. 1");
	cudaDeviceSynchronize(); // wait until kernel finishes and then come back to following code // not needed to check
	cudaMemcpy(host_a, dev_a, sizeof(int) * vectorSize, cudaMemcpyDeviceToHost);
	//check also here
	cudaMemcpy(host_b, dev_b, sizeof(int) * vectorSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_c, dev_c, sizeof(int) * vectorSize, cudaMemcpyDeviceToHost);

	printf("Execution 1: 1 block 64 threads \n");
	printArray(host_a, vectorSize, "threadIdx.x");
	printArray(host_b, vectorSize, "blockIdx.x");
	printArray(host_c, vectorSize, "globalId");

	grid.x = 64; // (64, 1, 1)
	block.x = 1; // (1, 1, 1)
	idKernel << < grid, block >> > (dev_a, dev_b, dev_c);
	cudaDeviceSynchronize();
	cudaMemcpy(host_a, dev_a, sizeof(int) * vectorSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_b, dev_b, sizeof(int) * vectorSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_c, dev_c, sizeof(int) * vectorSize, cudaMemcpyDeviceToHost);

	printf("\nExecution 2: 64 blocks 1 thread \n");
	printArray(host_a, vectorSize, "threadIdx.x");
	printArray(host_b, vectorSize, "blockIdx.x");
	printArray(host_c, vectorSize, "globalId");

	grid.x = 4;
	block.x = 16;
	idKernel << < grid, block >> > (dev_a, dev_b, dev_c);
	cudaDeviceSynchronize();
	cudaMemcpy(host_a, dev_a, sizeof(int) * vectorSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_b, dev_b, sizeof(int) * vectorSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_c, dev_c, sizeof(int) * vectorSize, cudaMemcpyDeviceToHost);

	printf("\nExecution 3: 4 block 16 threads \n");
	printArray(host_a, vectorSize, "threadIdx.x");
	printArray(host_b, vectorSize, "blockIdx.x");
	printArray(host_c, vectorSize, "globalId");

	free(host_a);
	free(host_b);
	free(host_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}
