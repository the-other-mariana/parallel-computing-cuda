
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


int main()
{
    int device = 0; // to store the number of devices we have
    int* count = &device;
    cudaGetDeviceCount(count); // needs a pointer to store the result
    // a device is a gpu card
    printf("Device count: %d\n", device);

    cudaDeviceProp properties;
    cudaDeviceProp* pProperties = &properties;
    cudaGetDeviceProperties(pProperties, device - 1); // device int is an index, we have one so index is zero
    printf("Name: %s\n", properties.name); // name of the device
    printf("multiProcessorCount: %d\n", properties.multiProcessorCount); 
    printf("maxBlocksPerMultiProcessor: %d\n", properties.maxBlocksPerMultiProcessor);
    // the sum of all the threads in each block
    printf("maxThreadsPerMultiProcessor: %d\n", properties.maxThreadsPerMultiProcessor);
    // max number of threads per block
    printf("maxThreadsPerBlock: %d\n", properties.maxThreadsPerBlock);

    // Grids dimensions
    printf("maxGridSize x axis: %d\n", properties.maxGridSize[0]); // max limit of blocks in x axis in the grid
    printf("maxGridSize y axis: %d\n", properties.maxGridSize[1]);
    printf("maxGridSize z axis: %d\n", properties.maxGridSize[2]);

    // Block dimensions (tweak but until the multip is <= 1024)
    printf("maxThreadsDim x axis: %d\n", properties.maxThreadsDim[0]); // max limit of threads per dimension in block
    printf("maxThreadsDim y axis: %d\n", properties.maxThreadsDim[1]);
    printf("maxThreadsDim z axis: %d\n", properties.maxThreadsDim[2]);

    return 0;
}

