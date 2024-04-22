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

void h_blur(unsigned char *arr, unsigned char *result) {
    int offset = 1 * width;
    for (int row = 1; row < height - 1; row++) {
        for (int col = 1; col < width - 1; col++) {
            int sum = 0;
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    sum += arr[(row + j) * width + (col + i)];
                }
            }
            result[offset + col] = sum / 9; // Total number of pixels in the neighborhood
        }
        offset += width;
    }
}

__global__ void d_blur(unsigned char *arr, unsigned char *result,
                       int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < 1 || col < 1 || row >= height - 1 || col >= width - 1)
        return;

    int sum = 0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int color = arr[(row + j) * width + (col + i)];
            sum += color;
        }
    }
    result[row * width + col] = sum / 9;
}

int main(int argc, char **argv) {
    unsigned char *d_resultPixels;
    unsigned char *h_resultPixels;
    unsigned char *h_pixels = NULL;
    unsigned char *d_pixels = NULL;

    char *srcPath = "sample_640426.pgm";
    char *h_ResultPath = "h_sample_640426.pgm";
    char *d_ResultPath = "d_sample_640426.pgm";

    loadPGMub(srcPath, &h_pixels, &width, &height);

    int ImageSize = sizeof(unsigned char) * width * height;

    h_resultPixels = (unsigned char *)malloc(ImageSize);
    cudaMalloc((void **)&d_pixels, ImageSize);
    cudaMalloc((void **)&d_resultPixels, ImageSize);
    cudaMemcpy(d_pixels, h_pixels, ImageSize, cudaMemcpyHostToDevice);

    clock_t starttime, endtime, difference;
    starttime = clock();

    // Apply mean blur on CPU
    h_blur(h_pixels, h_resultPixels);

    endtime = clock();
    difference = (endtime - starttime);
    double interval = difference / (double)CLOCKS_PER_SEC;
    printf("CPU execution time = %f ms\n", interval * 1000);
    savePGMub(h_ResultPath, h_resultPixels, width, height);

    dim3 block(16, 16);
    dim3 grid(width / 16, height / 16);
    unsigned int timer = 0;
    // cutCreateTimer(&timer); // Removed cutCreateTimer
    // cutStartTimer(timer); // Removed cutStartTimer

    // CUDA method
    d_blur<<<grid, block>>>(d_pixels, d_resultPixels, width, height);
    cudaDeviceSynchronize(); // Synchronize after kernel launch

    // cutStopTimer(timer); // Removed cutStopTimer
    // printf("CUDA execution time = %f ms\n", cutGetTimerValue(timer)); // Removed cutGetTimerValue

    cudaMemcpy(h_resultPixels, d_resultPixels, ImageSize, cudaMemcpyDeviceToHost);
    savePGMub(d_ResultPath, h_resultPixels, width, height);

    printf("Press enter to exit ...\n");
    getchar();
}
