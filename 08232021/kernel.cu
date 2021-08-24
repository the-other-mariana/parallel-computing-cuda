
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void printThreadIds()
{
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    printf("globalId: %d, threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d, blockDim.x: %d, blockIdx.x %d\n", globalId, threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockIdx.x);
}

__global__ void printGlobalId_oneBlockOneDim()
{
    printf("GlobalId: %d\n", threadIdx.x);
}

__global__ void printGlobalId_NBlocksOneDim()
{
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    printf("GlobalId: %d\n", globalId);
}

int main() {
    dim3 grid(3, 1, 1); 
    dim3 block(4, 1, 1);
    printThreadIds<< < grid, block >> > ();

    return 0;
}
