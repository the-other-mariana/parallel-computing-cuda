# Practice

## Exercise 1

- Function that solve a system of linear equations in the host.

- Function that solves a system of linear equations in the device through the launch of a kernel (1 block and 1 thread).

The kernel must receive all coefficients as a vector (of size 6).

A linear system with the form:

> ax + by = c <br />
> dx + ey = f,

Can be solved by the formulas:

> x = (ce - bf) / (ae -bd) <br />
> y = (af - cd) / (ae - bd)

```c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__host__ void linearSolveCPU(float* n, float* x, float* y) {
    *x = (n[2] * n[4] - n[1] * n[5]) / (n[0] * n[4] - n[1] * n[3]);
    *y = (n[0] * n[5] - n[2] * n[3]) / (n[0] * n[4] - n[1] * n[3]);
}

__global__ void linearSolveGPU(float* n, float* x, float* y)
{
    *x = (n[2] * n[4] - n[1] * n[5]) / (n[0] * n[4] - n[1] * n[3]);
    *y = (n[0] * n[5] - n[2] * n[3]) / (n[0] * n[4] - n[1] * n[3]);
}

int main()
{
    float* n_host = (float*)malloc(sizeof(float) * 6); // if malloc, you need to initialize all spaces one by one
    float* x_host = (float*)malloc(sizeof(float));
    float* y_host = (float*)malloc(sizeof(float));

    float* x_gpu = (float*)malloc(sizeof(float));
    float* y_gpu = (float*)malloc(sizeof(float));

    float* n_device;
    float* x_device;
    float* y_device;

    cudaMalloc((void**)&n_device, sizeof(float) * 6);
    cudaMalloc((void**)&x_device, sizeof(float));
    cudaMalloc((void**)&y_device, sizeof(float));

    n_host[0] = 5;
    n_host[1] = 1;
    n_host[2] = 4;
    n_host[3] = 2;
    n_host[4] = -3;
    n_host[5] = 5;

    *x_host = 0;
    *y_host = 0;
    *x_gpu = 0;
    *y_gpu = 0;

    cudaMemcpy(n_device, n_host, sizeof(float) * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(x_device, x_host, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, sizeof(float), cudaMemcpyHostToDevice);

    linearSolveCPU(n_host, x_host, y_host);
    printf("CPU result \n");
    printf("x = %f y = %f \n", *x_host, *y_host);

    linearSolveGPU <<< 1, 1 >>> (n_device, x_device, y_device);
    cudaMemcpy(x_gpu, x_device, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_gpu, y_device, sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU result \n");
    printf("x = %f y = %f \n", *x_gpu, *y_gpu);

    free(n_host);
    free(x_host);
    free(y_host);
    free(x_gpu);
    free(y_gpu);

    cudaFree(n_device);
    cudaFree(x_device);
    cudaFree(y_device);

    return 0;
}
```

## Lab 03

Make a program in c/c++ in which you launch a kernel with one block and one thread. The kernel must solve a quadratic equation in the form:

> ax^2 + bx + c = 0,

where its solutions are given by:

> x1 = (-b + sqrt(b^2 - 4ac)) / 2a <br />
> x2 = (-b - sqrt(b^2 - 4ac)) / 2a

For the implementation, you must consider:

1. Ask the user for coefficients a, b and c.
2. The program must show the solutions for the equation or a message stating that the solution does NOT exist if the result is an imaginary number.

### Tests

- a = 1, b = -5, c = 6 -> x1 = 2, x2 = 3

- a = 1, b = 1, c = 1 -> The solution does not exist


### Solution

```c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
    
}
```

## Other Findings

```c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<iostream>

using namespace std;

__global__ void solveGPU(double* dev_abc, double* dev_x1x2, int* dev_error)
{
    double root = (dev_abc[1] * dev_abc[1]) - (4 * dev_abc[0] * dev_abc[2]);
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
    double n_host[3] = { 0 };
    double x1x2_host[2] = { 0 };
    bool error_host = false;

    double* n_dev;
    double* x1x2_dev;
    int* error_dev; // gives no error
    cudaMalloc((void**)&n_dev, sizeof(double) * 3);
    cudaMalloc((void**)&x1x2_dev, sizeof(double) * 2);
    cudaMalloc((void**)&error_dev, sizeof(bool));

    for (int i = 0; i < 3; i++) {
        printf("%c: ", char(i + 97));
        scanf("%lf", &n_host[i]);
    }

    cudaMemcpy(n_dev, n_host, sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(x1x2_dev, x1x2_host, sizeof(double) * 2, cudaMemcpyHostToDevice); // not necessary
    cudaMemcpy(error_dev, &error_host, sizeof(bool), cudaMemcpyHostToDevice); // not necessary

    solveGPU << < 1, 1 >> > (n_dev, x1x2_dev, error_dev);


    cudaMemcpy(&error_host, error_dev, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(x1x2_host, x1x2_dev, sizeof(double) * 2, cudaMemcpyDeviceToHost);
    if (error_host) {
        printf("GPU Result:\n");
        printf("The solution does not exist\n");
    }
    else {
        printf("GPU Result:\n");
        printf("x1 = %lf x2 = %lf\n", x1x2_host[0], x1x2_host[1]);
    }

    //free(n_host); // exc
    //free(x1x2_host); // exc
    //free(&error_host); // exc

    cudaFree(n_dev);
    cudaFree(x1x2_dev);
    cudaFree(error_dev);
}

```

```c++
int* test;
cudaMalloc((void**)&test, sizeof(bool)); // no error
```

```c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<iostream>

using namespace std;

__global__ void solveGPU(double* dev_abc, double* dev_x1x2, int* dev_error)
{
    double root = (dev_abc[1] * dev_abc[1]) - (4 * dev_abc[0] * dev_abc[2]);
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
    double n_host[3] = { 0 }; 
    double x1x2_host[2] = { 0 };
    bool error_host = false;

    double* n_dev;
    double* x1x2_dev;
    int* error_dev; // gives no error
    cudaMalloc((void**)&n_dev, sizeof(double) * 3);
    cudaMalloc((void**)&x1x2_dev, sizeof(double) * 2);
    cudaMalloc((void**)&error_dev, sizeof(bool));

    for (int i = 0; i < 3; i++) {
        printf("%c: ", char(i + 97));
        scanf("%lf", &n_host[i]);
    }

    cudaMemcpy(n_dev, n_host, sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(x1x2_dev, x1x2_host, sizeof(double) * 2, cudaMemcpyHostToDevice); // not necessary
    cudaMemcpy(error_dev, &error_host, sizeof(bool), cudaMemcpyHostToDevice); // not necessary

    solveGPU << < 1, 1 >> > (n_dev, x1x2_dev, error_dev);


    cudaMemcpy(&error_host, error_dev, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(x1x2_host, x1x2_dev, sizeof(double) * 2, cudaMemcpyDeviceToHost);
    if (error_host) {
        printf("GPU Result:\n");
        printf("The solution does not exist\n");
    }
    else {
        printf("GPU Result:\n");
        printf("x1 = %lf x2 = %lf\n", x1x2_host[0], x1x2_host[1]);
    }
}
```

- `cudaMalloc(void** devPtr, size_t size)`: Allocates `size` bytes of linear memory on the device and returns in *devPtr a pointer to the allocated memory. Memory not cleared.

- `cudaMemcpy (void* dst, const void* src, size_t count, cudaMemcpyKind kind)`: Copies `count` bytes from the memory area pointed to by src to the memory area pointed to by dst. Calling cudaMemcpy() with dst and src pointers that do not match the direction of the copy results in an undefined behavior.