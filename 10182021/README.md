# Notes

## Kernel Validation

Verify just once that the kernel operations are correctly performed, which is crutial when working with large amounts of data. This consists in a comparison between the result of the same operation done both in CPU and GPU.

We would need to add a function that does that same pixel operation in the host, to continue the same examples with image processing, for example. This is basically making the kernel function but in the host.

Validation: execute the same kernel operations in the host and compare the results given by the CPU and GPU.

## Exercise

Execute a kernel validation for the last kernel where we made the complement of an RGB image. The kernel validation includes two functions: one for performing the complement in the CPU and one for comparing the CPU image and the GPU image.

### Solution

```c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__host__ void checkCUDAError(const char* msg) {
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), msg);
	}
}

__global__ void complement(uchar* RGB) {

	// locate my current block row
	int threads_per_block = blockDim.x * blockDim.y;
	int threads_per_row = threads_per_block * gridDim.x;
	int row_offset = threads_per_row * blockIdx.y;

	// locate my current block column
	int block_offset = blockIdx.x * threads_per_block;
	int threadId_inside = blockDim.x * threadIdx.y + threadIdx.x;

	// locate my current grid row
	int thread_per_grid = (gridDim.x * gridDim.y * threads_per_block);
	int gridOffset = blockIdx.z * thread_per_grid;

	int gId = gridOffset + row_offset + block_offset + threadId_inside;
	RGB[gId] = 255 - RGB[gId];
}

__host__ void complementCPU(Mat* original, Mat* comp) {
	for (int i = 0; i < original->rows; i++) {
		for (int j = 0; j < original->cols; j++) {
			comp->at<Vec3b>(i, j)[0] = 255 - original->at<Vec3b>(i, j)[0];
			comp->at<Vec3b>(i, j)[1] = 255 - original->at<Vec3b>(i, j)[1];
			comp->at<Vec3b>(i, j)[2] = 255 - original->at<Vec3b>(i, j)[2];
		}
	}
}

__host__ bool validationKernel(Mat img1, Mat img2) {
	Vec3b* pImg1, * pImg2;
	for (int k = 0; k < 3; k++) {
		for (int i = 0; i < img1.rows; i++) {
			pImg1 = img1.ptr<Vec3b>(i);
			pImg2 = img2.ptr<Vec3b>(i);
			for (int j = 0; j < img1.cols; j++) {
				if (pImg1[j][k] != pImg2[j][k]) {
					printf("Error at kernel validation\n");
					return true;
				}
			}
		}
	}
	printf("Kernel validation successful\n");
	return false;
}

int main() {

	Mat img = imread("antenaRGB.jpg");

	const int R = img.rows;
	const int C = img.cols;

	Mat imgComp(img.rows, img.cols, img.type());
	Mat imgCompCPU(img.rows, img.cols, img.type());
	uchar* host_rgb, * dev_rgb;
	host_rgb = (uchar*)malloc(sizeof(uchar) * R * C * 3);

	cudaMalloc((void**)&dev_rgb, sizeof(uchar) * R * C * 3);
	checkCUDAError("Error at malloc dev_r1");

	// matrix as vector
	for (int k = 0; k < 3; k++) {
		for (int i = 0; i < R; i++) {
			for (int j = 0; j < C; j++) {
				Vec3b pix = img.at<Vec3b>(i, j);

				host_rgb[i * C + j + (k * R * C)] = pix[k];

			}
		}
	}
	cudaMemcpy(dev_rgb, host_rgb, sizeof(uchar) * R * C * 3, cudaMemcpyHostToDevice);
	checkCUDAError("Error at memcpy host_rgb -> dev_rgb");

	dim3 block(32, 32);
	dim3 grid(C / 32, R / 32, 3);

	complement << < grid, block >> > (dev_rgb);
	cudaDeviceSynchronize();
	checkCUDAError("Error at kernel complement");

	cudaMemcpy(host_rgb, dev_rgb, sizeof(uchar) * R * C * 3, cudaMemcpyDeviceToHost);
	checkCUDAError("Error at memcpy host_rgb <- dev_rgb");

	for (int k = 0; k < 3; k++) {
		for (int i = 0; i < R; i++) {
			for (int j = 0; j < C; j++) {
				imgComp.at<Vec3b>(i, j)[k] = host_rgb[i * C + j + (k * R * C)];
			}
		}
	}

	complementCPU(&img, &imgCompCPU);
	bool error = validationKernel(imgCompCPU, imgComp);

	if (error) {
		printf("Check kernel operations\n");
		return 0;
	}


	imshow("Image", img);
	imshow("Image Complement CPU", imgCompCPU);
	imshow("Image Complement GPU", imgComp);
	waitKey(0);

	free(host_rgb);
	cudaFree(dev_rgb);

	return 0;
}

```