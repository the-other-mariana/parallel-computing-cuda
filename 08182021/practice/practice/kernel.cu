#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<iostream>

using namespace std;

__global__ void solveGPU(double* dev_abc, double* dev_x1x2, bool* dev_error)
{
    double root = (dev_abc[1] * dev_abc[1]) - (4 * dev_abc[0] * dev_abc[2]);
    // printf("root: %lf\n", root);
    if (root < 0) {
        *dev_error = true;
    }
    else {
        *dev_error = false;
        dev_x1x2[0] = ((-1 * dev_abc[1] - sqrt(root)) / (2 * dev_abc[0]));
        dev_x1x2[1] = ((-1 * dev_abc[1] + sqrt(root)) / (2 * dev_abc[0]));
    }

}

int main() {
    double* n_host = (double*)malloc(sizeof(double) * 3); // not cast, error
    double* x1x2_host = (double*)malloc(sizeof(double) * 2);
    bool* error_host = (bool*)malloc(sizeof(bool));

    double* n_dev;
    double* x1x2_dev;
    bool* error_dev;
    cudaMalloc((void**)&n_dev, sizeof(double) * 3);
    cudaMalloc((void**)&x1x2_dev, sizeof(double) * 2);
    cudaMalloc((void**)&error_dev, sizeof(bool)); // &bool error

    for (int i = 0; i < 3; i++) {
        printf("%c: ", char(i + 97)); //printf("%s", (i + 65)); exception
        scanf("%lf", &n_host[i]); // "A:%lf" not error, but input incomplete // \n weird results
    }

    x1x2_host[0] = 0;
    x1x2_host[1] = 0;
    *error_host = false;

    cudaMemcpy(n_dev, n_host, sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(x1x2_dev, x1x2_host, sizeof(double) * 2, cudaMemcpyHostToDevice); // not necessary
    cudaMemcpy(error_dev, error_host, sizeof(bool), cudaMemcpyHostToDevice); // not necessary

    solveGPU << < 1, 1 >> > (n_dev, x1x2_dev, error_dev);

    // cout << "cuda ptr " << *error_dev << endl; // no error, but execption at runtime
    cudaMemcpy(error_host, error_dev, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(x1x2_host, x1x2_dev, sizeof(double) * 2, cudaMemcpyDeviceToHost);
    if (*error_host) {
        printf("GPU Result:\n");
        printf("The solution does not exist\n");
    }
    else {
        printf("GPU Result:\n");
        printf("x1 = %lf x2 = %lf\n", x1x2_host[0], x1x2_host[1]);
    }

    const int x_const = 12;
    // error int* p3 = &x_const; 
    const int* p3 = &x_const; // no error, also this pointer can point to different const int variables
    //*p3 = 11; // error bc you're changing x_const or p3
    const int y_const = 10;
    p3 = &y_const; // p3 can change its target, but its variable type is const int

    /*
    char word[] = "hello!";
    char* p = word;
    char* p0 = &word[0];
    char* p3 = &word[3];

    cout << p << endl; // hello!
    cout << p0 << endl; // hello!
    cout << p3 << endl; // lo!

    int size;
    int* ptr;

    cout << "Enter size: ";
    cin >> size;

    ptr = (int*)malloc(sizeof(int) * size);

    for (int i = 0; i < size; i++) {
        cout << "Value: ";
        cin >> ptr[i];
    }

    for (int i = 0; i < size; i++) {
        cout << ptr[i] << " ";
    }
    cout << endl;

    int* ptr1;
    int* ptr2;

    ptr1 = (int*)malloc(10 * sizeof(int));
    *ptr1 = { 0 };
    for (int i = 0; i < 10; i++) {
        cout << ptr1[i] << " ";
    }
    // 0 -842150451 -842150451 -842150451 -842150451 -842150451 -842150451 -842150451 -842150451 -842150451
    cout << endl;
    
    for (int i = 0; i < 10; i++) {
        ptr1[i] = i * 10;
    }
    cout << *(ptr1 + 5) << endl; // 50
    */
}