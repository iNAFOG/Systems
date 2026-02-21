//Last time we wrote a linear kernel this time we will be writing a 2D kernel
//The core work we will be doing would be same - Multiplying by 2
//Writing a 2D kernel will help you with working when you will write kernels for images as some format as just 2D in nature
//It will also teach us how to handle threds in y dimensions with x dimensions.

#include<cuda_runtime.h>
#include<iostream>

__global__ void my2DKernel(float* input,int width,int height){
    //Computing 2D thread coordinates

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Only process valid threads
    if(x < width && y < height) {
        int idx = y * width + x;
        //its pretty standard way to do this
        input[idx] *= 2.0f;
    }
}

int main(){
    //Image dimentions
    const int width = 1024;
    const int height = 1024;
    const int N = width * height;

    float* h_image = nullptr;
    cudaMallocHost(&h_image, N * sizeof(float));

    for(int i = 0; i < N; ++i){
        h_image[i] = 1.0f;
    }

    float* d_image = nullptr;
    cudaMalloc(&d_image, N* sizeof(float));
    cudaMemcpy(d_image,h_image,N * sizeof(float),cudaMemcpyHostToDevice);

    //dim3 is a cuda struct useed to specify dimensions for Grid Size and Block size
    //so now it helps us to access the value in .x, .y .z dims, by deafault it gives value = 1 so .z  = 1 here

    dim3 threadsPerBlock2D(16,16);
    dim3 blocksPerGrid2D((width  + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,(height + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);

    //starting the kernel
    my2DKernel<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_image, width, height);
    

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_image, d_image, N * sizeof(float),cudaMemcpyDeviceToHost);

    // Verify a sample element
    std::cout << "h_image[0] = " << h_image[0] << std::endl; // should print 2.0

    // Cleanup
    cudaFree(d_image);
    cudaFreeHost(h_image);

    return 0;
}
