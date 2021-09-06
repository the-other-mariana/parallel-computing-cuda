
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


__global__ void arraySum(int* dev_a, int* dev_b, int* dev_c) {
    // 1block and this with one dimension
    int gId = threadIdx.x; // there will be 12 gId variables when all threads are executing the kernel
    // 12 vars, one for each thread
    dev_c[gId] = dev_a[gId] + dev_b[gId];
}

int main()
{
    const int vectorSize = 12;
    int host_a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    int host_b[] = { 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    int host_c[vectorSize] = { 0 };

    int* dev_a, * dev_b, * dev_c;
    cudaMalloc((void**)&dev_a, sizeof(int) * vectorSize);
    cudaMalloc((void**)&dev_b, sizeof(int) * vectorSize);
    cudaMalloc((void**)&dev_c, sizeof(int) * vectorSize);

    cudaMemcpy(dev_a, host_a, sizeof(int) * vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, sizeof(int) * vectorSize, cudaMemcpyHostToDevice);

    dim3 grid(1, 1, 1); // or dim3 grid(1);
    dim3 block(vectorSize, 1, 1); // or dim3 block(vectorSize);

    arraySum << < grid, block >> > (dev_a, dev_b, dev_c);

    cudaMemcpy(host_c, dev_c, sizeof(int) * vectorSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < vectorSize; i++) {
        printf("%d ", host_c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
