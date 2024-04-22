#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// Function to load a PGM file
void loadPGMub(const char *filename, unsigned char **pixels, unsigned int *width, unsigned int *height) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        exit(1);
    }

    // Read PGM header
    char magic[3];
    fscanf(file, "%s", magic);
    if (strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: Invalid PGM file format\n");
        fclose(file);
        exit(1);
    }

    fscanf(file, "%u", width);
    fscanf(file, "%u", height);
    unsigned int max_val;
    fscanf(file, "%u%*c", &max_val);

    // Allocate memory for image pixels
    *pixels = (unsigned char *)malloc((*width) * (*height) * sizeof(unsigned char));
    if (!*pixels) {
        fprintf(stderr, "Error: Unable to allocate memory\n");
        fclose(file);
        exit(1);
    }

    // Read pixel data
    fread(*pixels, sizeof(unsigned char), (*width) * (*height), file);
    fclose(file);
}

// Function to save a PGM file
void savePGMub(const char *filename, unsigned char *pixels, unsigned int width, unsigned int height) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Unable to create file %s\n", filename);
        exit(1);
    }

    // Write PGM header
    fprintf(file, "P5\n");
    fprintf(file, "%u %u\n", width, height);
    fprintf(file, "255\n");

    // Write pixel data
    fwrite(pixels, sizeof(unsigned char), width * height, file);
    fclose(file);
}

void h_EdgeDetect(unsigned char *org, unsigned char *result, unsigned int width, unsigned int height) {
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    int offset = width;

    for (int row = 1; row < height - 1; row++) {
        for (int col = 1; col < width - 1; col++) {
            int sumX = 0, sumY = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int curPixel = org[(row + j) * width + (col + i)];
                    sumX += curPixel * Gx[i + 1][j + 1];
                    sumY += curPixel * Gy[i + 1][j + 1];
                }
            }
            int sum = abs(sumX) + abs(sumY);
            if (sum > 255) sum = 255;
            if (sum < 0) sum = 0;
            result[offset + col] = sum;
        }
        offset += width;
    }
}

__global__ void d_EdgeDetect(unsigned char *org, unsigned char *result, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= 1 && col >= 1 && row < height - 1 && col < width - 1) {
        int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int Gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
        int sumX = 0, sumY = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int curPixel = org[(row + j) * width + (col + i)];
                sumX += curPixel * Gx[i + 1][j + 1];
                sumY += curPixel * Gy[i + 1][j + 1];
            }
        }
        int sum = abs(sumX) + abs(sumY);
        if (sum > 255) sum = 255;
        if (sum < 0) sum = 0;
        result[row * width + col] = sum;
    }
}

int main(int argc, char **argv) {
    unsigned char *d_resultPixels;
    unsigned char *h_resultPixels;
    unsigned char *h_pixels = NULL;
    unsigned char *d_pixels = NULL;
    unsigned int width, height;

    char *srcPath = "sample_640426.pgm";
    char *h_ResultPath = "h_sample_640426_sobel.pgm";
    char *d_ResultPath = "d_sample_640426_sobel.pgm";

    // Load the input image and get dimensions
    loadPGMub(srcPath, &h_pixels, &width, &height);

    int ImageSize = sizeof(unsigned char) * width * height;

    // Allocate memory for the results on host and device
    h_resultPixels = (unsigned char *)malloc(ImageSize);
    cudaMalloc((void **)&d_pixels, ImageSize);
    cudaMalloc((void **)&d_resultPixels, ImageSize);

    // Copy image data from host to device
    cudaMemcpy(d_pixels, h_pixels, ImageSize, cudaMemcpyHostToDevice);

    // Start timing the host (CPU) processing
    clock_t starttime = clock();
    h_EdgeDetect(h_pixels, h_resultPixels, width, height);
    clock_t endtime = clock();

    // Calculate the execution time for host processing
    double interval = (endtime - starttime) / (double)CLOCKS_PER_SEC;
    printf("CPU execution time = %f ms\n", interval * 1000);

    // Save the result of the host processing
    savePGMub(h_ResultPath, h_resultPixels, width, height);

    // Define block size and grid size for CUDA
    dim3 block(16, 16);
    dim3 grid(width/16, height/16);

    // Prepare to time the GPU processing
    float gpu_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Perform the edge detection on the device (GPU)
    d_EdgeDetect<<<grid, block>>>(d_pixels, d_resultPixels, width, height);
    cudaDeviceSynchronize();

    // Stop timing after synchronization
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU execution time = %f ms\n", gpu_time);


    // Copy the result from device to host
    cudaMemcpy(h_resultPixels, d_resultPixels, ImageSize, cudaMemcpyDeviceToHost);

    // Save the result of the device processing
    savePGMub(d_ResultPath, h_resultPixels, width, height);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_pixels);
    cudaFree(d_resultPixels);
    free(h_pixels);
    free(h_resultPixels);

    // Prompt to exit
    printf("Press enter to exit...\n");
    getchar();

    return 0;
}
