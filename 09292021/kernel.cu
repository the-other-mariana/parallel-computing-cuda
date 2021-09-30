
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;
__host__ void checkCUDAError(const char* msg) {
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), msg);
	}
}

int main()
{
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	// Device name: NVIDIA GeForce GTX 1080
	cout << "name: " << properties.name << endl;
	// 2560 Cores https://www.nvidia.com/es-la/geforce/products/10series/geforce-gtx-1080/
	cout << "CUDA Cores: " << 2560 << endl; 
	// SM units/ Multiprocessors: 
	cout << "multiProcessorCount: " << properties.multiProcessorCount << endl; // 20
	cout << "Cores per Multiprocessor: " << 2560 / properties.multiProcessorCount << endl; // 128
	cout << "maxThreadsPerMultiProcessor: " << properties.maxThreadsPerMultiProcessor << endl; // 2048
	cout << "maxBlocksPerMultiProcessor: " << properties.maxBlocksPerMultiProcessor << endl; // 32
	
	

    return 0;
}


