# Notes

- A warp is the basic unit of grouping threads. We make this concept because a warp is physically executed.

- A block with 33 threads has 2 warps.

- A thread is executed in a CUDA core and a CUDA core is a processing nucleus.

## Data Transfer and Memory Management

A Host (CPU) and Device (GPU) have their own separate memory. CUDA creates functions to transfer info from one memory to another to connect these memories.

**Global Memory** is a memory that is shared throughout all the blocks (and its threads) in a grid.

**Shared Memory** is an independent memory exclusive to the threads in the block that has the said shared memory.

Each thread has its own memory, called **Register**, which is very quick access but limited in space.

### Memory Management

The Host will reserve dynamic memory with `malloc(size)` and the Device will reserve memory in the Device's Global Memory using `cudaMalloc(void**, size)`.

### Syntax

```c++
void* malloc(size_t size);
cudaMalloc(void** devPtr, size_t size);

void free(void* _Block);
cudaFree(void* devPtr);
```

### Data Transfer

There are four types of data transfer:

Type of Transfer | Course Of Transfer | Description |
| ---- | ---- | ---- |
| cudaMemcpyHostToDevice | Host -> Device | Transfer data to be processed in parallel |
| cudaMemcpyDeviceToDevice | Device -> Device | Internal processing in the Device, without loops |
| cudaMemcpyDeviceToHost | Device -> Host | What was already processed in parallel, you return it to the Host to validate the info and see the results |
| cudaMemcpyHostToHost | Host -> Host | Normal CPU processing, but without loops |

### Syntax

```c++
cudaMemcpy(destination_mem, source_mem, size, typeOfTransfer);
```