#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

__host__ void checkCUDAError(const char* msg) {
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), msg);
	}
}

__global__ void complement(uchar* dev_a, uchar* dev_b) {
	// locate my current block row
	int threads_per_block = blockDim.x * blockDim.y;
	int threads_per_row = threads_per_block * gridDim.x;
	int row_offset = threads_per_row * blockIdx.y;

	// locate my current block column
	int block_offset = blockIdx.x * threads_per_block;
	int threadId_inside = blockDim.x * threadIdx.y + threadIdx.x;

	int gId = row_offset + block_offset + threadId_inside;
	dev_b[gId] = 255 - dev_a[gId];
}

using namespace cv;
int main() {

	Mat img = imread("antenaParalelo.jpg", IMREAD_GRAYSCALE);

	const int R = img.rows;
	const int C = img.cols;

	Mat imgResult(img.rows, img.cols, img.type());
	uchar* host_a, * host_b, * dev_a, * dev_b, * pImg;
	host_a = (uchar*)malloc(sizeof(uchar) * R * C);
	host_b = (uchar*)malloc(sizeof(uchar) * R * C);
	cudaMalloc((void**)&dev_a, sizeof(uchar) * R * C);
	checkCUDAError("Error at malloc dev_a");
	cudaMalloc((void**)&dev_b, sizeof(uchar) * R * C);
	checkCUDAError("Error at malloc dev_b");

	// matrix as vector
	for (int i = 0; i < R; i++) {
		pImg = img.ptr<uchar>(i); // points to a row each time
		for (int j = 0; j < C; j++) {
			host_a[i * C + j] = pImg[j];
		}
	}
	cudaMemcpy(dev_a, host_a, sizeof(uchar) * R * C, cudaMemcpyHostToDevice);

	dim3 block(32, 32);
	dim3 grid(C / 32, R / 32);

	complement << < grid, block >> > (dev_a, dev_b);
	checkCUDAError("Error at kernel");

	cudaMemcpy(host_b, dev_b, sizeof(uchar) * R * C, cudaMemcpyDeviceToHost);

	for (int i = 0; i < R; i++) {
		pImg = imgResult.ptr<uchar>(i); 
		for (int j = 0; j < C; j++) {
			pImg[j] = host_b[i * C + j];
		}
	}

	imshow("Image", img);
	imshow("Image Result", imgResult);
	waitKey(0); 

	free(host_a);
	free(host_b);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return 0;
}