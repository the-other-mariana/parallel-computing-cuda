#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__host__ void linearSolveCPU(float* n, float* x, float* y) {
    *x = (n[2] * n[3] - n[1] * n[5]) / (n[0] * n[4] - n[1] * n[3]);
    *y = (n[0] * n[5] - n[2] * n[3]) / (n[0] * n[4] - n[1] * n[3]);
}

__global__ void linearSolveGPU(float* n, float* x, float* y)
{
    *x = (n[2] * n[4] - n[1] * n[5]) / (n[0] * n[4] - n[1] * n[3]);
    *y = (n[0] * n[5] - n[2] * n[3]) / (n[0] * n[4] - n[1] * n[3]);
}

int main()
{
    float* n_host = (float*)malloc(sizeof(float) * 6); // if malloc, you need to initialize all spaces one by one
    float* x_host = (float*)malloc(sizeof(float));
    float* y_host = (float*)malloc(sizeof(float));

    float* x_gpu = (float*)malloc(sizeof(float));
    float* y_gpu = (float*)malloc(sizeof(float));

    float* n_device;
    float* x_device;
    float* y_device;

    cudaMalloc((void**)&n_device, sizeof(float) * 6);
    cudaMalloc((void**)&x_device, sizeof(float));
    cudaMalloc((void**)&y_device, sizeof(float));

    n_host[0] = 5;
    n_host[1] = 1;
    n_host[2] = 4;
    n_host[3] = 2;
    n_host[4] = -3;
    n_host[5] = 5;

    *x_host = 0;
    *y_host = 0;
    *x_gpu = 0;
    *y_gpu = 0;

    cudaMemcpy(n_device, n_host, sizeof(float) * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(x_device, x_host, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, sizeof(float), cudaMemcpyHostToDevice);

    linearSolveGPU << < 1, 1 >> > (n_device, x_device, y_device);
    cudaMemcpy(x_gpu, x_device, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_gpu, y_device, sizeof(float), cudaMemcpyDeviceToHost);
    printf("x = %f y = %f \n", *x_host, *y_host); // 0 and 0
    printf("GPU result \n");
    printf("x = %f y = %f \n", *x_gpu, *y_gpu);

    linearSolveCPU(n_host, x_host, y_host);
    printf("CPU result \n");
    printf("x = %f y = %f \n", *x_host, *y_host);

    free(n_host);
    free(x_host);
    free(y_host);
    free(x_gpu);
    free(y_gpu);

    cudaFree(n_device);
    cudaFree(x_device);
    cudaFree(y_device);

    return 0;
}