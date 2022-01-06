# Parallel Computing with Cuda

This repo contains notes and code related to learning parallel computing using Cuda. The compilation of all lecture notes and its corresponding code is in the [pdf file: Introduction to Parallel Computing with CUDA](https://github.com/the-other-mariana/pandoc-pdfs/blob/master/cuda01/book/cuda-book.pdf).

## Set Up

1. Follow the guide to install the Nvidia Cuda Development Toolkit, from https://docs.nvidia.com/cuda/.

2. Find your Nvidia GPU model at: https://developer.nvidia.com/cuda-gpus#compute and search for its computing capability.

> In my case, my GPU is GeForce GTX 960M, so its computing capability is 5.0, which will be needed later.

### Linux (Console)

The complete history of commands to install it is in the [linux-cmds.txt](https://github.com/the-other-mariana/parallel-computing-cuda/blob/master/cuda-cmds.txt) file. <br />

![image](https://github.com/the-other-mariana/parallel-computing-cuda/blob/master/media/linux-setup.jpeg?raw=true) <br />

Provide the computing capability as `compute_XX,sm_XX` in flag form when running the command nvcc to compile a .cu code.

### Windows (Visual Studio 2019)

Provide the computing capability as `compute_XX,sm_XX` in Visual Studio Project Properties > CUDA > Device > Generated Code menu.

1. Create a CUDA Runtime project in Visual Studio (2019).

2. On the project explorer menu, right click and go to Project Properties.

![image](https://github.com/the-other-mariana/parallel-computing-cuda/blob/master/media/win-setup-01.png?raw=true) <br />

3. Go to CUDA > Device and click on Edit on the Code Generation field.

4. Write `compute_XX,sm_XX`, where the XX is replaced by your GPU computing capability, and then uncheck the Inherit checkbox.

![image](https://github.com/the-other-mariana/parallel-computing-cuda/blob/master/media/win-setup-02.png?raw=true) <br />

5. Click on OK and debug (run) your code.

## Handy Links

- http://mylifeismymessage.net/find-the-compute-capability-of-your-nvidia-graphics-card-gpu/

- https://askubuntu.com/questions/965499/lspci-grep-i-nvidia-command-not-working-to-check-for-cuda-compatability

- https://stackoverflow.com/questions/28384986/kernel-seem-not-to-execute

- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions
