# Notes

## The Kernel

It is a method executed in the GPU as a mass execution. As seen before, CUDA architecure's Processing flow is done switching between CPU and GPU. The kernel is a function you execute on the Device.

Specifier | Called From | Executed In | Syntax |
| ---- | ---- | ---- | ---- |
| \_\_host\_\_ | Host | Host | \_\_float\_\_ float name() |
| \_\_global\_\_ | Host | Device | \_\_global\_\_ void name() |
| \_\_device\_\_ | Device | Device | \_\_device\_\_ float name() |

- If there is no specifier before a function, it is simply taken as a normal function in CPU processing. In this way, `__host__` is just a simple CPU function as well.

- `__device__` functions are defined throughout your code and then a kernel function calls it. Could be or not a parallel process function, it is just a method you need in your kernel to be done.

- A kernel return value always void. If you want to return something, you do it by reference using the kernel parameters.

- The specifier `__global__` creates a kernel.

## The Kernel Syntax

The kernel call is done in the CPU, the execution is in GPU.

```c++
__global__ void myKernel(arg_1, arg_2, ..., arg_n) {
    // code to be executed in the GPU
}

// from CPU you call the kernel
myKernel<<<blocks,threads>>>(arg_1, arg_2, ..., arg_n);
```

### Exercise 1

```c++
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
    cudaMalloc((void**)&dev_num1, sizeof(int));
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
```

At the line `int* host_num1 = (int*)malloc(sizeof(int));`, Visual Studio makes a suggestion, which is the same I put in the comment on that line. <br />

![image](https://github.com/the-other-mariana/parallel-computing-cuda/blob/master/08162021/alt01.png?raw=true) <br />