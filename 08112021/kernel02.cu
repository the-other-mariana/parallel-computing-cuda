
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

int main()
{
    int N = 8;
    int* host_vectorA;
    int* host_vectorB;
    int* host_vectorC;

    int* device_vectorA;
    int* device_vectorB;
    int* device_vectorC;

    // reserve memory in the Host
    host_vectorA = (int*)malloc(sizeof(int) * N);
    host_vectorB = (int*)malloc(sizeof(int) * N);
    host_vectorC = (int*)malloc(sizeof(int) * N);

    // reserve memory in the Device
    cudaMalloc((void**)&device_vectorA, sizeof(int) * N);
    cudaMalloc((void**)&device_vectorB, sizeof(int) * N);
    
    for (int i = 0; i < N; i++) {
        host_vectorA[i] = i;
    }

    // data transfer: copy the host array to the device array (two different arrays)
    cudaMemcpy(device_vectorA, host_vectorA, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_vectorB, device_vectorA, sizeof(int) * N, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_vectorB, device_vectorB, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_vectorC, host_vectorB, sizeof(int) * N, cudaMemcpyHostToHost);

    for (int i = 0; i < N; i++) {
        printf("%d ", host_vectorC[i]);
    }

    // free memory
    free(host_vectorA);
    free(host_vectorB);
    free(host_vectorC);
    cudaFree(device_vectorA);
    cudaFree(device_vectorB);

    return 0;
}

