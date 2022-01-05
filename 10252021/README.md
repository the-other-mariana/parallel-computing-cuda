# Configs Cheatsheet

## 1D Grid & 1D Block

![img](res/1.png)

- Block along x axis

```c++
int gId = threadIdx.x;
```

![img](res/2.png)

- Block along y axis

```c++
int gId = threadIdx.y;
```

## 1D Grid & N 1D Blocks

![img](res/3.png)

- N 1D blocks along x axis

```c++
int threadsPerBlock = blockDim.x;
int blockOffset = threadsPerBlock * blockIdx.x;
int idInsideBlock = threadIdx.x;
int gId = blockOffset + idInsideBlock;
```

## 1D Grid & N 2D Blocks

![img](res/4.png)

- N 2D blocks along x axis

```c++
int threadsPerBlock = blockDim.x * blockDim.y;
int blockOffset = threadsPerBlock * blockIdx.x;
int idInsideBlock = blockDim.x * threadIdx.y + threadIdx.x;
int gId = blockOffset + idInsideBlock;
```

## 1D Grid & N 1D Blocks

![img](res/5.png)

- N 1D blocks along y axis

```c++
int threadsPerBlock = blockDim.x;
int rowOffset = threadsPerBlock * blockIdx.y;
int idInsideBlock = threadIdx.x;
int gId = rowOffset + idInsideBlock;
```

## 1D Grid & N 1D Blocks

![img](res/6.png)

- 1D grid along x and 1D blocks along its y

```c++
int threadsPerBlock = blockDim.y;
int blockOffset = threadsPerBlock * blockIdx.x;
int idInsideBlock = threadIdx.y;
int gId = blockOffset + idInsideBlock;
```

## 2D Grid & 2D Blocks

![img](res/7.png)

- 2D grid (x and y) and 2D blocks (x and y)

```c++
int threadsPerBlock = blockDim.x * blockDim.y;
int threadsPerRow = threadsPerBlock * gridDim.x;
int rowOffset = threadsPerRow * blockIdx.y;
int blockOffset = threadsPerBlock * blockIdx.x;
int idInsideBlock = blockDim.x * threadIdx.y + threadIdx.x;
int gId = rowOffset + blockOffset + idInsideBlock;
```

## 3D Grid & 2D Blocks

![img](res/8.png)

- dim3 grid(3, 4, 3) and dim3 block(32, 32, 1)

```c++
int threadsPerBlock = blockDim.x * blockDim.y;
int threadsPerRow = threadsPerBlock * gridDim.x;
int rowOffset = threadsPerRow * blockIdx.y;
int blockOffset = threadsPerBlock * blockIdx.x;
int idInsideBlock = blockDim.x * threadIdx.y + threadIdx.x;
int threadsPerGrid = threadsPerBlock * gridDim.x * gridDim.y;
int gridOffset = threadsPerGrid * blockIdx.z;
int gId = gridOffset + rowOffset + blockOffset + idInsideBlock;
```

### Exercises

What would be the gId formula for the configs below?

![img](res/9.png)