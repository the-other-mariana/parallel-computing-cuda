# Notes

## Error Management in CUDA

CUDA provides a way to handle errors that involve **exceeded GPU capacities** or **GPU malfunctioning**. These are not logic nor syntax errors.

- `cudaError_t error;` is a CUDA type that is really an integer, and this number gives us a hint. Every CUDA function returns an error thta can be stored in this type. The only CUDA funtcion thta doesnt return a `cudaError_t` variable is a kernel itself: it must return void.

```c++
cudaError_t error;
error = cudaMalloc((void**)&ptr, size);
```

## Types of Errors

- `cudaSuccess = 0`: successful execution

- `cudaErrorInvalidValue = 1`: This indicates that one or more parameters passed to the API function call is not within an acceptable range of values. An enum param in a CUDA function that you didnt match, or a different data type sent.

- `cudaErrorMemoryAllocation = 2`: the API call failed because it was unable to allocate enough memory to perform the operation. When you do not have enough space in memory: you would normally write on memory outside of your array, but if there is no more mem left, this happens.

- `cudaErrorInvalidConfiguration = 9`: When you exceed the max num of blocks or threads of the GPU card.

- `cudaErrorInvalidMemcpyDirection = 21`: the direction of the memcpy passed to API call is not one of the types specified by cudaMemcpyKind. You put another word, basically.

## Process

```c++
cudaError_t error;
error = cudaMalloc((void**)&ptr, size);
cudaGetErrorString(error);

// after a kernel launch
error = cudaGetLastError();
cudaGetErrorString(error);
```

- `error` would be an intger, but in order to avoid checking in the docs, CUDA provides the function `cudaGetErrorString(error)` that, given an integer, it returns a string of the error details/explanation.

- To get a kernel error (otherwise a kernel is just void return), we can catch the last integer of error with `cudaGetLastError();`.

Open any project an type this funtion that will be used after any CUDA function call:

```c++
// host because every function call must be from host
__host__ void checkCUDAError(const char* msg){
    cudaError_t error;
    cudaDeviceSynchronize(); // avoid catching another error that is not the next we want
    error = cudaGetLastError(); // status of the last CUDA API call (maybe 0 or success, not error)
    if (error != cudaSuccess){
        printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), msg);
    }
}
```

Because the host/device execution is asynchronous (both at the same time), we need to take care of the sequence and sometimes you need to **pause** and wait for the kernel to finish in order to come back to the host: we need to synchronize.

### Output

```
ERROR 9: invalid configuration argument (Error at idKernel execution no. 1)
Execution 1: 1 block 64 threads
threadIdx.x: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
blockIdx.x: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
globalId: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

Execution 2: 64 blocks 1 thread
threadIdx.x: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
blockIdx.x: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
globalId: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63

Execution 3: 4 block 16 threads
threadIdx.x: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
blockIdx.x: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
globalId: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
```

*ERROR 9: Kernel no. 1 is not really executed*