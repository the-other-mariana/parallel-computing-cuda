#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void solveGPU(double* dev_abc, double* dev_x1x2, bool* dev_error)
{
    double root = (dev_abc[1] * dev_abc[1]) - (4 * dev_abc[0] * dev_abc[2]);
    // printf("root: %lf\n", root);
    if (root < 0) {
        *dev_error = true;
    }
    else {
        *dev_error = false;
        dev_x1x2[0] = ((-1 * dev_abc[1] - sqrt(root)) / (2 * dev_abc[0]));
        dev_x1x2[1] = ((-1 * dev_abc[1] + sqrt(root)) / (2 * dev_abc[0]));
    }
     
}

int main() {
    double* n_host = (double*)malloc(sizeof(double) * 3);
    double* x1x2_host = (double*)malloc(sizeof(double) * 2);
    bool* error_host = (bool*)malloc(sizeof(bool));

    double* n_dev;
    double* x1x2_dev;
    bool* error_dev;
    cudaMalloc((void**)&n_dev, sizeof(double) * 3);
    cudaMalloc((void**)&x1x2_dev, sizeof(double) * 2);
    cudaMalloc((void**)&error_dev, sizeof(bool));

    for (int i = 0; i < 3; i++) {
        scanf("%lf", &n_host[i]);
    }

    x1x2_host[0] = 0;
    x1x2_host[1] = 0;
    *error_host = false;

    cudaMemcpy(n_dev, n_host, sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(x1x2_dev, x1x2_host, sizeof(double) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(error_dev, error_host, sizeof(bool), cudaMemcpyHostToDevice);
    
    solveGPU <<< 1, 1 >>> (n_dev, x1x2_dev, error_dev);

    cudaMemcpy(error_host, error_dev, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(x1x2_host, x1x2_dev, sizeof(double) * 2, cudaMemcpyDeviceToHost);
    if (*error_host) {
        printf("GPU Result:\n");
        printf("The solution does not exist\n");
    }
    else {
        printf("GPU Result:\n");
        printf("x1 = %lf x2 = %lf\n", x1x2_host[0], x1x2_host[1]);
    }
    
}