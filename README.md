# Parallel Computing with Cuda

This repo contains notes and code related to learning parallel computing using Cuda.

## Set Up

1. Follow the guide to install the Nvidia Cuda Development Toolkit.

2. Find your Nvidia GPU model at: http://mylifeismymessage.net/find-the-compute-capability-of-your-nvidia-graphics-card-gpu/ and serach for its computing capability.

> In my case, my GPU is GeForce 960M, so its computing capability is 5.0, which will be needed later.

### Linux

The complete history of commands to install it is in the [linux-cmds.txt](https://github.com/the-other-mariana/parallel-computing-cuda/blob/master/cuda-cmds.txt) file. <br />

![image](https://github.com/the-other-mariana/parallel-computing-cuda/blob/master/media/linux-setup.jpeg?raw=true) <br />

Provide the computing capability as `compute_XX, sm_XX` in flag form when running the command nvcc to compile a .cu code.

### Windows

Provide the computing capability as `compute_XX, sm_XX` in Visual Studio Project Properties > CUDA > Device > Generated Code menu.

![image](https://github.com/the-other-mariana/parallel-computing-cuda/blob/master/media/win-setup.jpeg?raw=true) <br />
