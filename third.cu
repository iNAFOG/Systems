//CUDA
//Let's start with what is CUDA?
//CUDA is technically an extension of C/C++ which let's us run code on a GPU
//The compiler required to run and compiler cuda code is NVCC
//The extenstion of cuda file is .cu


//In CUDA a kernel is a special function which runs on gpu, to write a kernel we start with the special keyword in start of that function.
//__global__ any function starting with that is executed on GPU in CUDA code. 

//<<< >>> These are called chevrons - they are used to specify total block and threads for the kernel to run.

__global__ void kernel() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    kernel<<<1, 5>>>();   // Launch kernel with 1 block, 5 threads

    cudaDeviceSynchronize(); // Wait for GPU to finish
    return 0;
}