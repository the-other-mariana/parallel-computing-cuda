#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

__global__ void complement(uchar* b, uchar* g, uchar* r) {
    int threads_in_a_block = blockDim.x * blockDim.y; // 4
    int threads_in_a_row = threads_in_a_block * gridDim.x; // 12
    int row_offset = threads_in_a_row * blockIdx.y; // 12

    int block_offset = blockIdx.x * threads_in_a_block; // 4
    int threadId_in_a_block = blockDim.x * threadIdx.y + threadIdx.x; // 1
    int gId = row_offset + block_offset + threadId_in_a_block; // 12 + 4 + 1

    uchar b_value = b[gId], g_value = g[gId], r_value = r[gId];

    b[gId] = 255 - b_value;
    g[gId] = 255 - g_value;
    r[gId] = 255 - r_value;
}

__global__ void contrast(uchar* b, uchar* g, uchar* r, float fc) {
    int threads_in_a_block = blockDim.x * blockDim.y;
    int threads_in_a_row = threads_in_a_block * gridDim.x;
    int row_offset = threads_in_a_row * blockIdx.y;

    int block_offset = blockIdx.x * threads_in_a_block;
    int threadId_in_a_block = blockDim.x * threadIdx.y + threadIdx.x;
    int gId = row_offset + block_offset + threadId_in_a_block;

    int b_value = fc * b[gId], g_value = fc * g[gId], r_value = fc * r[gId];

    if (b_value > 255) b_value = 255;
    if (b_value < 0) b_value = 0;

    if (g_value > 255) g_value = 255;
    if (g_value < 0) g_value = 0;

    if (r_value > 255) r_value = 255;
    if (r_value < 0) r_value = 0;

    b[gId] = b_value;
    g[gId] = g_value;
    r[gId] = r_value;
}

__global__ void brightness(uchar* b, uchar* g, uchar* r, float fb) {
    int threads_in_a_block = blockDim.x * blockDim.y;
    int threads_in_a_row = threads_in_a_block * gridDim.x;
    int row_offset = threads_in_a_row * blockIdx.y;

    int block_offset = blockIdx.x * threads_in_a_block;
    int threadId_in_a_block = blockDim.x * threadIdx.y + threadIdx.x;
    int gId = row_offset + block_offset + threadId_in_a_block;

    int b_value = fb + b[gId], g_value = fb + g[gId], r_value = fb + r[gId];

    if (b_value > 255) b_value = 255;
    if (b_value < 0) b_value = 0;

    if (g_value > 255) g_value = 255;
    if (g_value < 0) g_value = 0;

    if (r_value > 255) r_value = 255;
    if (r_value < 0) r_value = 0;

    b[gId] = b_value;
    g[gId] = g_value;
    r[gId] = r_value;
}

__host__ void check_CUDA_Error(const char* mensaje) {
    cudaError_t error;
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("ERROR %d, %s (%s)\n", error, cudaGetErrorString(error), mensaje);
    }
}

__host__ void fill_img(Mat img, uchar* b, uchar* g, uchar* r) {
    Vec3b* p_img;
    int index;

    for (int i = 0; i < img.rows; i++) {
        p_img = img.ptr<Vec3b>(i);
        for (int j = 0; j < img.cols; j++) {
            index = j + i * img.cols;
            p_img[j][0] = b[index];
            p_img[j][1] = g[index];
            p_img[j][2] = r[index];
        }
    }
}

__host__ void read_data(Mat img, uchar* b, uchar* g, uchar* r) {
    Vec3b* p_img;
    int index = 0;

    for (int i = 0; i < img.rows; i++) {
        p_img = img.ptr<Vec3b>(i); // fila i
        for (int j = 0; j < img.cols; j++) {
            index = j + i * img.cols;
            b[index] = p_img[j][0];
            g[index] = p_img[j][1];
            r[index] = p_img[j][2];
        }
    }
}

__host__ void host_to_device(uchar* host_b, uchar* host_g, uchar* host_r, uchar* dev_b, uchar* dev_g, uchar* dev_r, int SIZE) {
    cudaMemcpy(dev_b, host_b, SIZE, cudaMemcpyHostToDevice);
    check_CUDA_Error("mecpy hostToDevice b");
    cudaMemcpy(dev_g, host_g, SIZE, cudaMemcpyHostToDevice);
    check_CUDA_Error("mecpy hostToDevice g");
    cudaMemcpy(dev_r, host_r, SIZE, cudaMemcpyHostToDevice);
    check_CUDA_Error("mecpy hostToDevice r");
}

__host__ void device_to_host(uchar* host_b, uchar* host_g, uchar* host_r, uchar* dev_b, uchar* dev_g, uchar* dev_r, int SIZE) {
    cudaMemcpy(host_b, dev_b, SIZE, cudaMemcpyDeviceToHost);
    check_CUDA_Error("mecpy deviceToHost b");
    cudaMemcpy(host_g, dev_g, SIZE, cudaMemcpyDeviceToHost);
    check_CUDA_Error("mecpy deviceToHost g");
    cudaMemcpy(host_r, dev_r, SIZE, cudaMemcpyDeviceToHost);
    check_CUDA_Error("mecpy deviceToHost r");
}

int main() {
    // leer imagen
    Mat img = imread("antenaRGB.jpg");

    if (!img.data) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // dimensiones
    const int N = img.rows;
    const int M = img.cols;
    const int SIZE = N * M * sizeof(uchar);

    // declaración
    uchar* host_b, * host_g, * host_r;
    uchar* dev_b, * dev_g, * dev_r;
    Vec3b* p_img;

    // resultado
    Mat img_complement(img.rows, img.cols, img.type());
    Mat img_contrast(img.rows, img.cols, img.type());
    Mat img_brightness(img.rows, img.cols, img.type());

    // reservación de memoria
    host_b = (uchar*)malloc(SIZE);
    host_g = (uchar*)malloc(SIZE);
    host_r = (uchar*)malloc(SIZE);

    cudaMalloc((void**)&dev_b, SIZE);
    check_CUDA_Error("malloc dev_b");
    cudaMalloc((void**)&dev_g, SIZE);
    check_CUDA_Error("malloc dev_g");
    cudaMalloc((void**)&dev_r, SIZE);
    check_CUDA_Error("malloc dev_r");

    // llenar datos
    read_data(img, host_b, host_g, host_r);

    // configuración
    dim3 grid(M / 32, N / 32);
    dim3 block(32, 32);

    // -------------- COMPLEMENTO -------------------------
    // copiar a device
    host_to_device(host_b, host_g, host_r, dev_b, dev_g, dev_r, SIZE);

    // lanzamiento de kernel
    complement << <grid, block >> > (dev_b, dev_g, dev_r);
    cudaDeviceSynchronize();
    device_to_host(host_b, host_g, host_r, dev_b, dev_g, dev_r, SIZE);

    // regresar a formato matricial
    fill_img(img_complement, host_b, host_g, host_r);

    // ----------------------------------------------------

    // ---------------- CONTRASTE -------------------------
    read_data(img, host_b, host_g, host_r);

    host_to_device(host_b, host_g, host_r, dev_b, dev_g, dev_r, SIZE);

    contrast << <grid, block >> > (dev_b, dev_g, dev_r, 1.5);
    cudaDeviceSynchronize();
    device_to_host(host_b, host_g, host_r, dev_b, dev_g, dev_r, SIZE);

    fill_img(img_contrast, host_b, host_g, host_r);
    // ----------------------------------------------------

    // ------------------ BRILLO --------------------------
    read_data(img, host_b, host_g, host_r);

    host_to_device(host_b, host_g, host_r, dev_b, dev_g, dev_r, SIZE);

    brightness << <grid, block >> > (dev_b, dev_g, dev_r, -100.0);
    cudaDeviceSynchronize();
    device_to_host(host_b, host_g, host_r, dev_b, dev_g, dev_r, SIZE);

    fill_img(img_brightness, host_b, host_g, host_r);
    // ----------------------------------------------------

    imshow("img", img);
    imshow("complement", img_complement);
    imshow("contrast", img_contrast);
    imshow("brightness", img_brightness);

    free(host_b);
    free(host_g);
    free(host_r);

    cudaFree(dev_b);
    cudaFree(dev_g);
    cudaFree(dev_r);

    waitKey(0);

    return 0;
}
