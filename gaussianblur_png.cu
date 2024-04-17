#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
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

    if (col < width && row < height) {
        int pixelIndex = (row * width + col) * 3;
        result[pixelIndex] = getPixel(arr, col, row, width);  // Red
        result[pixelIndex + 1] = getPixel(arr, col, row, width);  // Green
        result[pixelIndex + 2] = getPixel(arr, col, row, width);  // Blue
    }
}

int main() {
    // Define image path
    const char *inputImagePath = "Sample-png-image-100kb.png";
    const char *outputImagePath = "output.png";

    // Load PNG image using OpenCV
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        printf("Error: Unable to load input image\n");
        return EXIT_FAILURE;
    }

    // Allocate memory on device and copy input image data
    size_t imageSize = inputImage.rows * inputImage.cols * inputImage.channels();
    unsigned char *d_imageData;
    cudaMalloc((void **)&d_imageData, imageSize);
    cudaMemcpy(d_imageData, inputImage.data, imageSize, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((inputImage.cols + block.x - 1) / block.x, (inputImage.rows + block.y - 1) / block.y);

    // Call kernel to apply blur
    d_blur<<<grid, block>>>(d_imageData, d_imageData, inputImage.cols, inputImage.rows);

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
        cudaFree(d_imageData);
        return EXIT_FAILURE;
    }

    // Copy result back to host
    cudaMemcpy(inputImage.data, d_imageData, imageSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_imageData);

    // Save modified image
    if (!cv::imwrite(outputImagePath, inputImage)) {
        printf("Error: Unable to save output image\n");
        return EXIT_FAILURE;
    }

    printf("Image processing complete\n");

    return EXIT_SUCCESS;
}
