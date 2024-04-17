#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define MASK_WIDTH 3
#define MASK_HEIGHT 3

__constant__ int mask[MASK_HEIGHT][MASK_WIDTH] = {
    {1, 2, 1},
    {2, 3, 2},
    {1, 2, 1}
};

__device__ int getPixel(unsigned char *arr, int col, int row, int imgWidth) {
    int sum = 0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int color = arr[(row + j) * imgWidth + (col + i)];
            sum += color * mask[i + 1][j + 1];
        }
    }
    return sum / 15;
}

__global__ void d_blur(unsigned char *arr, unsigned char *result,
                       int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < 2 || col < 2 || row >= height - 3 || col >= width - 3)
        return;

    result[row * width + col] = getPixel(arr, col, row, width);
}

int main(int argc, char **argv) {
    FILE *file;
    unsigned char *h_pixels, *h_resultPixels;
    unsigned char *d_pixels, *d_resultPixels;
    char *srcPath = "image.pgm";
    char *h_ResultPath = "output_cpu.pgm";
    char *d_ResultPath = "output_gpu.pgm";
    unsigned int width, height, imageSize;

    // Load PGM file
    file = fopen(srcPath, "rb");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s\n", srcPath);
        return EXIT_FAILURE;
    }
    fscanf(file, "P5\n%d %d\n255\n", &width, &height);
    imageSize = width * height;
    h_pixels = (unsigned char *)malloc(imageSize);
    h_resultPixels = (unsigned char *)malloc(imageSize);
    fread(h_pixels, sizeof(unsigned char), imageSize, file);
    fclose(file);

    // Allocate memory on device
    cudaMalloc((void **)&d_pixels, imageSize);
    cudaMalloc((void **)&d_resultPixels, imageSize);

    // Copy input data to device
    cudaMemcpy(d_pixels, h_pixels, imageSize, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Start timing
    clock_t starttime = clock();

    // Call kernel
    d_blur<<<grid, block>>>(d_pixels, d_resultPixels, width, height);

    // Synchronize
    cudaDeviceSynchronize();

    // End timing
    clock_t endtime = clock();
    double interval = (double)(endtime - starttime) / CLOCKS_PER_SEC * 1000;

    // Copy result back to host
    cudaMemcpy(h_resultPixels, d_resultPixels, imageSize, cudaMemcpyDeviceToHost);

    // Save output image
    file = fopen(d_ResultPath, "wb");
    fprintf(file, "P5\n%d %d\n255\n", width, height);
    fwrite(h_resultPixels, sizeof(unsigned char), imageSize, file);
    fclose(file);

    printf("CUDA execution time = %f ms\n", interval);

    // Free memory
    free(h_pixels);
    free(h_resultPixels);
    cudaFree(d_pixels);
    cudaFree(d_resultPixels);

    return EXIT_SUCCESS;
}
