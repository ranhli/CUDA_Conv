#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <chrono>
#include <iostream>

using namespace std;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_SIZE 16
#define NUM_SLICE 3

__global__ void conv(uint8_t *inputImage, uint8_t *outputImage, int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Define kernel coefficients
        float kernel[5][5] = {
            {0.003765, 0.015019, 0.023792, 0.015019, 0.003765},
            {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
            {0.023792, 0.094907, 0.150342, 0.094907, 0.023792},
            {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
            {0.003765, 0.015019, 0.023792, 0.015019, 0.003765}
        };

        // Apply convolution
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0;
            for (int i = -2; i <= 2; ++i) {
                for (int j = -2; j <= 2; ++j) {
                    int curRow = row + i;
                    int curCol = col + j;
                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                        int index = (curRow * width + curCol) * channels + c;
                        sum += inputImage[index] * kernel[i + 2][j + 2];
                    }
                }
            }
            int index = (row * width + col) * channels + c;
            outputImage[index] = (uint8_t)sum;
        }
    }
}

int main() {
    const char *filename = "data/1-low-res.png";
    const char *outputFilename = "output/1-low-res-out-cu.png";

    int width, height, channels;
    uint8_t *inputImage = stbi_load(filename, &width, &height, &channels, 0);
    if (!inputImage) {
        fprintf(stderr, "Error loading image.\n");
        return 1;
    }

    int slice_height = height / NUM_SLICE;
    
    size_t slice_image_size = width * slice_height * channels * sizeof(uint8_t);
    size_t full_image_size = width * height * channels * sizeof(uint8_t);

    printf("Full image size: %zd\n", full_image_size);
    printf("Slice image size: %zd\n", slice_image_size);
    
    uint8_t *outputImage = (uint8_t *)malloc(full_image_size);
    
    uint8_t *d_inputImage, *d_outputImage;
    cudaMalloc((void **)&d_inputImage, slice_image_size);
    cudaMalloc((void **)&d_outputImage, slice_image_size);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (slice_height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_SLICE; ++i) {
        int offset = i * slice_height * width * channels;
        cudaMemcpy(d_inputImage, inputImage + offset, slice_image_size, cudaMemcpyHostToDevice);
        conv<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, slice_height, channels);
        cudaDeviceSynchronize();
        cudaMemcpy(outputImage + offset, d_outputImage, slice_image_size, cudaMemcpyDeviceToHost);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Convolution time: " << elapsed.count() << " seconds\n";

    stbi_write_png(outputFilename, width, height, channels, outputImage, width * channels);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    stbi_image_free(inputImage);
    free(outputImage);

    return 0;
}

