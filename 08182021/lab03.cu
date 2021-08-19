
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void linearSolveGPU(float* dev_abc, float* dev_x1x2, bool* dev_error)
{
    
}

int main()
{
    float* n_host = (float*)malloc(sizeof(float) * 3); 
    float* x1x2_host = (float*)malloc(sizeof(float) * 2);
    float* x1x2_gpu = (float*)malloc(sizeof(float) * 2);
    bool* error = (bool*)malloc(sizeof(bool));

    float* n_device;
    float* x1x2_device;

    cudaMalloc((void**)&n_device, sizeof(float) * 3);
    cudaMalloc((void**)&x1x2_device, sizeof(float) * 2);

    n_host[0] = 5;
    n_host[1] = 1;
    n_host[2] = 4;

    x1x2_host[0] = 0;
    x1x2_host[1] = 0;
    x1x2_gpu[0] = 0;
    x1x2_gpu[1] = 0;

    cudaMemcpy(n_device, n_host, sizeof(float) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(x1x2_device, x1x2_host, sizeof(float) * 2, cudaMemcpyHostToDevice);

    linearSolveGPU <<< 1, 1 >>> (n_device, x1x2_device, y_device);
    cudaMemcpy(x1x2_gpu, x1x2_device, sizeof(float) * 2, cudaMemcpyDeviceToHost);
    printf("GPU result \n");
    printf("x = %f y = %f \n", x1x2_gpu[0], x1x2_gpu[1]);

    free(n_host);
    free(x1x2_host);
    free(x1x2_gpu);

    cudaFree(n_device);
    cudaFree(x1x2_device);

    return 0;
}