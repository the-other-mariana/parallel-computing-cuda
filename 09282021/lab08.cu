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

__global__ void warpDetails() {
	int gId = blockIdx.y * (gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x) + threadIdx.x;
	int warpId = threadIdx.x / 32; // index of warp per block, not unique
	int gBlockId = blockIdx.y * gridDim.x + blockIdx.x;
	printf("threadIdx.x: %d blockIdx.x: %d blockIdx.y: %d gId: %d warpId: %d gBlockId: %d\n", threadIdx.x, blockIdx.x, blockIdx.y, gId, warpId, gBlockId);
}

int main() {
	dim3 block(42);
	dim3 grid(2, 2);
	warpDetails << < grid, block >> > ();
	checkCUDAError("Error at kernel");

	return 0;
}