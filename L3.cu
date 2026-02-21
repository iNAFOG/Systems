//CUDA
//Let's start with what is CUDA?
//CUDA is technically an extension of C/C++ which let's us run code on a GPU
//The compiler required to run and compiler cuda code is NVCC
//The extenstion of cuda file is .cu


//In CUDA a kernel is a special function which runs on gpu, to write a kernel we start with the special keyword in start of that function.
//__global__ any function starting with that is executed on GPU in CUDA code. 

//<<< >>> These are called chevrons - they are used to specify total blocks and threads for the kernel to run.
// my_first_kernel.cu
//------------------------Code:
// We will be for initial writing a simple kernel which doubles the number.

#include <cstdio>
//for printf and scanf functions
#include <cuda_runtime.h>
//provides access to CUDA fxns

// Kernel: myKernel running on the device (GPU)
// input : device pointer to float array of length N
// N     : total number of elements in the input

__global__ void myKernel(float* input, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //when we launch a kernel we want a way to identify which thread is the kernel running
    //above is very common way to do it
    // blockIdx.x -> Index of the current block executing
    // blockDim.x -> Size of the current block executing
    // threadIdx.x -> Current thread executing

    //.x means in the x dimention - more on it as we go on writing complex kernels

    //This is because if GPU lauches more threads then required (which in most cases is true)
    //So we do no processing on the extra threads
    //One of the most common checking lines you will see in kernels

    if (idx < N) {
        input[idx] *= 2.0f;   
        //Also pointer arithmetic here input[idx] means same as *(input+idx)
        //We are derefencing the value and then multiply by 2.
    }
}

int main()
{
    // 1) Problem size: one million floats
    const int N = 1'000'000;

    float* h_input = nullptr; //float pointer
    float* d_input = nullptr; //float pointer

    //quick revision - A pointer is variable which stores the memory address of another variable

    // 2) Allocate pinned host memory
    cudaMallocHost(&h_input, N * sizeof(float));
    //allocates the memory block in host(cpu) memory
    //h_input holds the memory address of the first block of memory
    //the memory allocated using malloc is in a continuous block
    
    // 3) Initialize host data
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }

    // 4) Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    //same as host in step 2 but now on gpu(device memory)

    // 5) Copy host → device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 6) Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 7) Launch kernel
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);

    // 8) Wait for GPU to finish
    cudaDeviceSynchronize();

    // 9) Copy results device → host
    cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Optional: print first element to verify
    printf("h_input[0] = %f\n", h_input[0]);  // should be 2.0f

    // Cleanup
    cudaFree(d_input);
    cudaFreeHost(h_input);

    return 0;
}


