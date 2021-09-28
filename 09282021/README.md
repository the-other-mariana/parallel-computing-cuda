# Notes

- There is a property that shows you the size of the warp: a warp is constituted of 32 consecutive threads. These threads use the SIMT modality (Single Instruction Multiple Threads) when executed: this means that each thread will execute the same instructions but with different data, its **own** variables or memory. Each thread will be executed in a streaming processor (SP) or CUDA Core or Nucleus. Each block is executed in a Streaming Multiprocessor, which has 32 small squares (SP) / 1 warp. 

- Only 32(SP per SM) x 8(SM's) = 256 threads will be run in real parallelism. Otherwise, there will be some waiting time.

- A warp is a basic unit that will help us with the decision of which block config to use. Each SM is divided into warps. 



- Each of those 128 threads are divided into warps: 4 warps. Each block will be divided into 4, and what will happen is that each block of 32 threads will be taken and so on, until you run 1024 threads in total.

    - You need 4 warps to execute a 128 thread block: a SM can only run 1 warp at the same time. If you config a warp per SM (or more, depending on how many cores the SM has), all SM's will run its warp in parallel and avoid the waiting time.

    - An SM has a number of threads, which will be grouped in warps. The SM runs all its threads (warps) in parallel. In this examples, we are saying that an SM runs only 32 threads or 1 warp in parallel.

    - Warps vontinue the gIds from the past warp executed: each thread will have its global id following the last id from the last thread in the last warp executed.

- The block config affects the execution time: it is advised that the block config is always a multiple of 32 (warp): you can loose time and also waste threads launched because of other configs.

## Lab 09

### Solution

```c++
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
```

### Output

![img](https://github.com/the-other-mariana/parallel-computing-cuda/blob/master/09282021/out-lab08.png?raw=true)

