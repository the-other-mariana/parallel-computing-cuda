# Notes

## Constant Memory

Apart from all the other memories shown in the previous diagram, we have Constant Memory:

![img](res/1.PNG)

The difference with this memory is that it is a **read-only** memory access, thus it is used from data that we just need to read. This memory is inside the Device, but its reservation it's done outside a kernel:

```c++
#define N 32
__constant__ int dev_A[N*N];
__global__ void kernel(int* dev_A, int* dev_B){

}
int main(){

}
```

Before using this constant memory, we used to send the information stored in global memory (*devPtr) as a parameter dev_A, and another vector for the results dev_B. Now, we can replace the reservation of dev_A in the Global Memory. After tne reservation, we used `cudaMemcpy()` to copy the data to dev_A. We also can forget about this step, and instead we transfer the data from the host to dev_A using `cudaMemcpyToSymbol(dev_A, host_A, sizeof(int)*N*N)`: this new function does not require the direction of the transference.

Constant Memory is advised to be used when we only have read-only data, because the memory access is faster than Global Memory.

## Implementation

Given a square matrix of size N, output the transpose of such matrix using Constant Memory. 

![img](res/2.PNG)

Generate a vector in the host: `[1,2,3,4,5,6,7,8,9]`, and copy them to the constant memory using `cudaMemcpyToSymbol(dev_A, host_A, sizeof(int)*N*N)`, now in dev_A. The result will be stored in dev_B as `[1,4,7,2,5,8,3,6,9]`. We do not need to send dev_A as parameter to the kernel, only parameter dev_B is sent. Thus, the result is in Global Memory, in dev_B. Therefore, we only need `cudaMalloc` for dev_B.

- 1 1D grid

- 1 2D block

```c++
dim3 grid(1);
dim3 block(N, N);
```

- Use validation for the kernel