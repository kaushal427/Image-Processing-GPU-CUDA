#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

void loadPGMub(const char *filename, unsigned char **pixels, unsigned int *width, unsigned int *height) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        exit(1);
    }

    char magic[3];
    fscanf(file, "%s", magic);
    if (strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: Invalid PGM file format\n");
        fclose(file);
        exit(1);
    }

    fscanf(file, "%u %u", width, height);
    unsigned int max_val;
    fscanf(file, "%u%*c", &max_val);

    *pixels = (unsigned char *)malloc((*width) * (*height) * sizeof(unsigned char));
    if (!*pixels) {
        fprintf(stderr, "Error: Unable to allocate memory\n");
        fclose(file);
        exit(1);
    }

    fread(*pixels, sizeof(unsigned char), (*width) * (*height), file);
    fclose(file);
}

void savePGMub(const char *filename, unsigned char *pixels, unsigned int width, unsigned int height) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Unable to create file %s\n", filename);
        exit(1);
    }

    fprintf(file, "P5\n");
    fprintf(file, "%u %u\n", width, height);
    fprintf(file, "255\n");
    fwrite(pixels, sizeof(unsigned char), width * height, file);
    fclose(file);
}

// Compute the Gaussian blur of a pixel
int getPixel(unsigned char *arr, int col, int row, int width, int height) {
    int mask[7][7] = {
        {1, 1, 2, 2, 2, 1, 1},
        {1, 2, 2, 4, 2, 2, 1},
        {2, 2, 4, 8, 4, 2, 2},
        {2, 4, 8, 16, 8, 4, 2},
        {2, 2, 4, 8, 4, 2, 2},
        {1, 2, 2, 4, 2, 2, 1},
        {1, 1, 2, 2, 2, 1, 1}
    };

    int sum = 0, count = 0;
    for (int j = -3; j <= 3; j++) {
        for (int i = -3; i <= 3; i++) {
            int newY = row + j;
            int newX = col + i;
            if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                sum += arr[newY * width + newX] * mask[i + 3][j + 3];
                count += mask[i + 3][j + 3];
            }
        }
    }
    return sum / count;
}

// Apply Gaussian blur on the whole image
void h_blur(unsigned char *arr, unsigned char *result, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            result[row * width + col] = getPixel(arr, col, row, width, height);
        }
    }
}

__global__ void d_blur(unsigned char *arr, unsigned char *result, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height)
        return;

    int mask[7][7] = {
        {1, 1, 2, 2, 2, 1, 1},
        {1, 2, 2, 4, 2, 2, 1},
        {2, 2, 4, 8, 4, 2, 2},
        {2, 4, 8, 16, 8, 4, 2},
        {2, 2, 4, 8, 4, 2, 2},
        {1, 2, 2, 4, 2, 2, 1},
        {1, 1, 2, 2, 2, 1, 1}
    };



    int sum = 0, count = 0;
    for (int j = -3; j <= 3; j++) {
        for (int i = -3; i <= 3; i++) {
            int newY = row + j;
            int newX = col + i;
            if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                sum += arr[newY * width + newX] * mask[i + 3][j + 3];
                count += mask[i + 3][j + 3];
            }
        }
    }
    result[row * width + col] = sum / count;
}

int main(int argc, char **argv) {
    unsigned char *d_resultPixels;
    unsigned char *h_resultPixels;
    unsigned char *h_pixels = NULL;
    unsigned char *d_pixels = NULL;
    unsigned int width, height;

    const char* imagePaths[] = {"mountains.pgm", "nature.pgm", "tower.pgm"};
    int numImages = sizeof(imagePaths) / sizeof(char*);

    for (int imgIndex = 0; imgIndex < numImages; imgIndex++) {
        char srcPath[100];
        char h_ResultPath[120];
        char d_ResultPath[120];

        sprintf(srcPath, "%s", imagePaths[imgIndex]);
        sprintf(h_ResultPath, "h_%s_gaussian.pgm", srcPath);
        sprintf(d_ResultPath, "d_%s_gaussian.pgm", srcPath);

        loadPGMub(srcPath, &h_pixels, &width, &height);

        int ImageSize = sizeof(unsigned char) * width * height;

        h_resultPixels = (unsigned char *)malloc(ImageSize);
        cudaMalloc((void **)&d_pixels, ImageSize);
        cudaMalloc((void **)&d_resultPixels, ImageSize);
        cudaMemcpy(d_pixels, h_pixels, ImageSize, cudaMemcpyHostToDevice);

        clock_t starttime = clock();
        h_blur(h_pixels, h_resultPixels, width, height);
        clock_t endtime = clock();
        double interval = (endtime - starttime) / (double)CLOCKS_PER_SEC;
        printf("CPU execution time = %f ms for %s\n", interval * 1000, srcPath);
        savePGMub(h_ResultPath, h_resultPixels, width, height);

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        float gpu_time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        d_blur<<<grid, block>>>(d_pixels, d_resultPixels, width, height);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time, start, stop);
        printf("GPU execution time = %f ms for %s\n", gpu_time, srcPath);

        cudaMemcpy(h_resultPixels, d_resultPixels, ImageSize, cudaMemcpyDeviceToHost);
        savePGMub(d_ResultPath, h_resultPixels, width, height);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_pixels);
        cudaFree(d_resultPixels);
        free(h_pixels);
        free(h_resultPixels);
    }

    printf("Press enter to exit ...\n");
    getchar();

    return 0;
}
