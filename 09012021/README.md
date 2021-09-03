# Notes

- 1 block and this block has only 1 dimension: globalId = threadIdx.x

- The kernel is in charge of mass processing. The kernel function will be executed in parallel N times, through the N threads. We need to identify the thread in order to give the thread different data to perform its kernel code.