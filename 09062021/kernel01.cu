#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h> /* srand, rand */
#include <time.h> /* time */
__global__ void sumaVectores(int* vecA, int* vecB, int* vecRes) {
	int gId = threadIdx.x + blockDim.x * blockIdx.x;
	vecRes[gId] = vecA[gId] + vecB[gId];
}
int main()
{
	const int vectorSize = 32;
	int* host_a = (int*)malloc(sizeof(int) * vectorSize);
	int* host_b = (int*)malloc(sizeof(int) * vectorSize);
	int* host_c = (int*)malloc(sizeof(int) * vectorSize);

	int* dev_a, * dev_b, * dev_c;

	cudaMalloc((void**)&dev_a, sizeof(int) * vectorSize);
	cudaMalloc((void**)&dev_b, sizeof(int) * vectorSize);
	cudaMalloc((void**)&dev_c, sizeof(int) * vectorSize);

	srand(time(NULL));

	for (int i = 0; i < vectorSize; i++) {
		int num = rand() % vectorSize + 1;
		host_a[i] = num;
		num = rand() % vectorSize + 1;
		host_b[i] = num;
	}

	cudaMemcpy(dev_a, host_a, sizeof(int) * vectorSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, sizeof(int) * vectorSize, cudaMemcpyHostToDevice);
	dim3 grid(4, 1, 1);
	dim3 block(8, 1, 1);

	sumaVectores << < grid, block >> > (dev_a, dev_b, dev_c);

	cudaMemcpy(host_c, dev_c, sizeof(int) * vectorSize, cudaMemcpyDeviceToHost);

	printf("Vector A: \n");
	for (int i = 0; i < vectorSize; i++) {
		printf("%d ", host_a[i]);
	}
	printf("\nVector B: \n");
	for (int i = 0; i < vectorSize; i++) {
		printf("%d ", host_b[i]);
	}
	printf("\nVector C: \n");
	for (int i = 0; i < vectorSize; i++) {
		printf("%d ", host_c[i]);
	}

	free(host_a);
	free(host_b);
	free(host_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}