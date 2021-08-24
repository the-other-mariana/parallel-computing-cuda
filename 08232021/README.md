# Notes

## Many Threads, One Block

- Threads (orange cubes) are contained or grouped in blocks (yellow container). In the image, there are six threads, one block. In this way, The kernel (function) will be executed by each thread at the same time (in parallel). This means that if we were to launch a kernel with this config (six threads one block), all we coded inside the kernel would be executed six times, one by each thread.

![img](https://github.com/the-other-mariana/parallel-computing-cuda/blob/master/08232021/res/01-simple-block.png?raw=true)

Blocks are tridimensional, as well as grids. The above block is one dimensional.

- To identify threads:

    - `threadIdx.x`: x axis index number of the thread. If you have only one block with six threads along one axis (one dimensional block), this property will be the full id. Indexes start in zero. It's y and z components are zero in one dimensional blocks, like the following image. 

    ![img](https://github.com/the-other-mariana/parallel-computing-cuda/blob/master/08232021/res/02-id-x.png?raw=true) 

    In the case of **a one dimensional block**, a thread would be identified with only one property: `threadIdx.x`. Properties `threadIdx.y` and `threadIdx.z` are zero. For example, (3,0,0).

    - `threadIdx.y`: you will also need y component if your block is bidimensional, like the below image. Z component is zero in that case as well.

    ![img](https://github.com/the-other-mariana/parallel-computing-cuda/blob/master/08232021/res/03-2d-block.png?raw=true)

    In the case of **a two dimensional block**, to identify a thread in this single-block config, you would need two components: `threadIdx.x` and `threadIdx.y`, while `threadIdx.z` is zero. For example (4,1,0).

    - `threadIdx.z`: for the case of **a three dimensional block**, you will need a third component to identify threads in it, called `threadIdx.z`, which indicates the position of the thread in the z axis. For example, (4,0,1).

    ![img](https://github.com/the-other-mariana/parallel-computing-cuda/blob/master/08232021/res/04-3d-block.png?raw=true)

    We need to identify threads in order to give instructions to particular threads inside the kernel.

- globalID: is the id that allows me to identify a thread from others. In the case of one dimensional blocks, `globalID = threadIdx.x`.

- `dim3` objects have the properties x, y and z. We can determine the configuration (number of blocks per axis) of the grid using the constructor `dim3 grid(3,1,1)`, for example. You can create a dim3 object to configure the dimensions of the blocks (threads quantity in each axis or direction). Therefore,

    - `dim3 grid`: how blocks are organized in the grid.
    - `dim3 block`: how threads are organized in the blocks.

### Image 2

```c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void printThreadIds()
{
    printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
    dim3 grid(1, 1, 1); 
    dim3 block(6, 2, 1);
    printThreadIds << < grid, block >> > ();

    return 0;
}
```

Its output is:

```
threadIdx.x: 0, threadIdx.y: 0, threadIdx.z: 0
threadIdx.x: 1, threadIdx.y: 0, threadIdx.z: 0
threadIdx.x: 2, threadIdx.y: 0, threadIdx.z: 0
threadIdx.x: 3, threadIdx.y: 0, threadIdx.z: 0
threadIdx.x: 4, threadIdx.y: 0, threadIdx.z: 0
threadIdx.x: 5, threadIdx.y: 0, threadIdx.z: 0
threadIdx.x: 0, threadIdx.y: 1, threadIdx.z: 0
threadIdx.x: 1, threadIdx.y: 1, threadIdx.z: 0
threadIdx.x: 2, threadIdx.y: 1, threadIdx.z: 0
threadIdx.x: 3, threadIdx.y: 1, threadIdx.z: 0
threadIdx.x: 4, threadIdx.y: 1, threadIdx.z: 0
threadIdx.x: 5, threadIdx.y: 1, threadIdx.z: 0
```

### Image 3

```c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void printThreadIds()
{
    printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
    dim3 grid(1, 1, 1); 
    dim3 block(6, 2, 3);
    printThreadIds << < grid, block >> > ();

    return 0;
}
```

Its output is:

```
threadIdx.x: 2, threadIdx.y: 1, threadIdx.z: 2
threadIdx.x: 3, threadIdx.y: 1, threadIdx.z: 2
threadIdx.x: 4, threadIdx.y: 1, threadIdx.z: 2
threadIdx.x: 5, threadIdx.y: 1, threadIdx.z: 2
threadIdx.x: 0, threadIdx.y: 0, threadIdx.z: 0
threadIdx.x: 1, threadIdx.y: 0, threadIdx.z: 0
threadIdx.x: 2, threadIdx.y: 0, threadIdx.z: 0
threadIdx.x: 3, threadIdx.y: 0, threadIdx.z: 0
threadIdx.x: 4, threadIdx.y: 0, threadIdx.z: 0
threadIdx.x: 5, threadIdx.y: 0, threadIdx.z: 0
threadIdx.x: 0, threadIdx.y: 1, threadIdx.z: 0
threadIdx.x: 1, threadIdx.y: 1, threadIdx.z: 0
threadIdx.x: 2, threadIdx.y: 1, threadIdx.z: 0
threadIdx.x: 3, threadIdx.y: 1, threadIdx.z: 0
threadIdx.x: 4, threadIdx.y: 1, threadIdx.z: 0
threadIdx.x: 5, threadIdx.y: 1, threadIdx.z: 0
threadIdx.x: 0, threadIdx.y: 0, threadIdx.z: 1
threadIdx.x: 1, threadIdx.y: 0, threadIdx.z: 1
threadIdx.x: 2, threadIdx.y: 0, threadIdx.z: 1
threadIdx.x: 3, threadIdx.y: 0, threadIdx.z: 1
threadIdx.x: 4, threadIdx.y: 0, threadIdx.z: 1
threadIdx.x: 5, threadIdx.y: 0, threadIdx.z: 1
threadIdx.x: 0, threadIdx.y: 1, threadIdx.z: 1
threadIdx.x: 1, threadIdx.y: 1, threadIdx.z: 1
threadIdx.x: 2, threadIdx.y: 1, threadIdx.z: 1
threadIdx.x: 3, threadIdx.y: 1, threadIdx.z: 1
threadIdx.x: 4, threadIdx.y: 1, threadIdx.z: 1
threadIdx.x: 5, threadIdx.y: 1, threadIdx.z: 1
threadIdx.x: 0, threadIdx.y: 0, threadIdx.z: 2
threadIdx.x: 1, threadIdx.y: 0, threadIdx.z: 2
threadIdx.x: 2, threadIdx.y: 0, threadIdx.z: 2
threadIdx.x: 3, threadIdx.y: 0, threadIdx.z: 2
threadIdx.x: 4, threadIdx.y: 0, threadIdx.z: 2
threadIdx.x: 5, threadIdx.y: 0, threadIdx.z: 2
threadIdx.x: 0, threadIdx.y: 1, threadIdx.z: 2
threadIdx.x: 1, threadIdx.y: 1, threadIdx.z: 2
```

Because of the repeated Ids, we have a *unique* Id for each thread in a block, called **globalId**.

### One Block, One Dimension

```c++
__global__ void printGlobalId_oneBlockOneDim()
{
    printf("GlobalId: %d\n", threadIdx.x);
}

int main() {
    dim3 grid(1, 1, 1); 
    dim3 block(6, 1, 1);
    printGlobalId_oneBlockOneDim << < grid, block >> > ();

    return 0;
}
```

### N blocks in X axis

Then,

> int globalId = threadIdx.x + blockDim.x * blockIdx.x;

```c++
__global__ void printGlobalId_NBlocksOneDim()
{
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    printf("GlobalId: %d\n", globalId);
}

int main() {
    dim3 grid(3, 1, 1); 
    dim3 block(3, 1, 1);
    printGlobalId_NBlocksOneDim<< < grid, block >> > ();

    return 0;
}
```

Which outputs: 

```
GlobalId: 6
GlobalId: 7
GlobalId: 8
GlobalId: 3
GlobalId: 4
GlobalId: 5
GlobalId: 0
GlobalId: 1
GlobalId: 2
```

```c++
__global__ void printThreadIds()
{
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;
    printf("globalId: %d, threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d, blockDim.x: %d, blockIdx.x %d\n", globalId, threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockIdx.x);
}
int main() {
    dim3 grid(3, 1, 1); 
    dim3 block(4, 1, 1);
    printThreadIds<< < grid, block >> > ();

    return 0;
}
```

Which outputs (threads (threadIdx.x) per block is ordered): 

```
globalId: 0, threadIdx.x: 0, threadIdx.y: 0, threadIdx.z: 0, blockDim.x: 4, blockIdx.x 0
globalId: 1, threadIdx.x: 1, threadIdx.y: 0, threadIdx.z: 0, blockDim.x: 4, blockIdx.x 0
globalId: 2, threadIdx.x: 2, threadIdx.y: 0, threadIdx.z: 0, blockDim.x: 4, blockIdx.x 0
globalId: 3, threadIdx.x: 3, threadIdx.y: 0, threadIdx.z: 0, blockDim.x: 4, blockIdx.x 0
globalId: 8, threadIdx.x: 0, threadIdx.y: 0, threadIdx.z: 0, blockDim.x: 4, blockIdx.x 2
globalId: 9, threadIdx.x: 1, threadIdx.y: 0, threadIdx.z: 0, blockDim.x: 4, blockIdx.x 2
globalId: 10, threadIdx.x: 2, threadIdx.y: 0, threadIdx.z: 0, blockDim.x: 4, blockIdx.x 2
globalId: 11, threadIdx.x: 3, threadIdx.y: 0, threadIdx.z: 0, blockDim.x: 4, blockIdx.x 2
globalId: 4, threadIdx.x: 0, threadIdx.y: 0, threadIdx.z: 0, blockDim.x: 4, blockIdx.x 1
globalId: 5, threadIdx.x: 1, threadIdx.y: 0, threadIdx.z: 0, blockDim.x: 4, blockIdx.x 1
globalId: 6, threadIdx.x: 2, threadIdx.y: 0, threadIdx.z: 0, blockDim.x: 4, blockIdx.x 1
globalId: 7, threadIdx.x: 3, threadIdx.y: 0, threadIdx.z: 0, blockDim.x: 4, blockIdx.x 1
```

Homework: ONLY GPU.