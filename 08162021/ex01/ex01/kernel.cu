#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__host__ int addCPU(int* num1, int* num2) {
    return(*num1 + *num2);
}

// kernel: __global__
__global__ void addGPU(int* num1, int* num2, int* res)
{
    *res = *num1 + *num2;
}

int main()
{
    // reserve mem in host
    int* host_num1 = (int*)malloc(sizeof(int)); // could be a simple integer and then you pass as param the &variable
    int* host_num2 = (int*)malloc(sizeof(int));
    int* host_resCPU = (int*)malloc(sizeof(int));
    int* host_resGPU = (int*)malloc(sizeof(int));

    // reserve mem in dev
    int* dev_num1, * dev_num2, * dev_res;
    cudaMalloc((void**)&dev_num1, sizeof(int)); // &3 error // &intvar no error but you need pointers with malloc in cuda
    cudaMalloc((void**)&dev_num2, sizeof(int));
    cudaMalloc((void**)&dev_res, sizeof(int)); // this pointer points to an address in the device

    // init data
    *host_num1 = 2;
    *host_num2 = 3;
    *host_resCPU = 0;
    *host_resGPU = 0;

    // data transfer
    cudaMemcpy(dev_num1, host_num1, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_num2, host_num2, sizeof(int), cudaMemcpyHostToDevice);

    // CPU call to CPU func
    *host_resCPU = addCPU(host_num1, host_num2);
    printf("CPU result \n");
    printf("%d + %d = %d \n", *host_num1, *host_num2, *host_resCPU);

    // CPU call to GPU func
    addGPU <<< 1, 1 >>> (dev_num1, dev_num2, dev_res);
    // dev_res is a pointer made with cudaMalloc (Global Memory)
    cudaMemcpy(host_resGPU, dev_res, sizeof(int), cudaMemcpyDeviceToHost);
    printf("GPU result \n");
    // dev_num1 is án address in GPU, you cannot access it from CPU
    printf("%d + %d = %d \n", *host_num1, *host_num2, *host_resGPU);

    // free memory
    free(host_num1);
    free(host_num2);
    free(host_resCPU);
    free(host_resGPU);

    cudaFree(dev_num1);
    cudaFree(dev_num2);
    cudaFree(dev_res);

    return 0;
}