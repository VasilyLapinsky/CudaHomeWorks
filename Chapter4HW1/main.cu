#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"

#include <iostream>

const size_t NUMBER_0F_POINTS = 1024 * 1024;
const size_t THREADS_PER_BLOCK = 1024;
const size_t BLOCKS_PER_GRID = std::min(size_t(32), (NUMBER_0F_POINTS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

curandGenerator_t CreateCuRandGenerator();
size_t CountPointsInTheCircle(float* randomX, float* randomY);

int main()
{
    // initialize clocks 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Allocate data
    float* randomX;
    float* randomY;
    cudaMalloc((void**)&randomX, NUMBER_0F_POINTS * sizeof(float));
    cudaMalloc((void**)&randomY, NUMBER_0F_POINTS * sizeof(float));

    // Generate random points
    curandGenerator_t gen = CreateCuRandGenerator();
    curandGenerateUniform(gen, randomX, NUMBER_0F_POINTS);
    curandGenerateUniform(gen, randomY, NUMBER_0F_POINTS);
    // claculates points in the circle
    size_t pointsInTheCircle = CountPointsInTheCircle(randomX, randomY);
    // calculates pi
    float pi = (4.f * static_cast<float>(pointsInTheCircle)) / static_cast<float>(NUMBER_0F_POINTS);

    std::cout << "Result pi: " << pi << '\n';

    // Free data in the end of the programm
    cudaFree(randomX);
    cudaFree(randomY);
    // calculate execution time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float   elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed time: " << elapsedTime << " ms\n";
    return 0;
}

curandGenerator_t CreateCuRandGenerator()
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    return gen;
}

__global__ void CalculatePointsIntheCircle(float* randomX, float* randomY, size_t* result)
{
    __shared__ size_t cache[THREADS_PER_BLOCK];
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t cacheIndex = threadIdx.x;

    size_t temp = 0;
    float x, y;
    while (tid < NUMBER_0F_POINTS) {
        x = randomX[tid];
        y = randomY[tid];
        temp += sqrt(x * x + y * y) < 1.f ? 1 : 0;
        tid += blockDim.x * gridDim.x;
    }

    // set the cache values
    cache[cacheIndex] = temp;
    __syncthreads();

    // reduction
    size_t i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
        {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
    {
        result[blockIdx.x] = cache[0];
    }
    __syncthreads();
}

size_t CountPointsInTheCircle(float* randomX, float* randomY)
{
    size_t* resultCountsDevice;
    cudaMalloc((void**)&resultCountsDevice, BLOCKS_PER_GRID * sizeof(size_t));

    CalculatePointsIntheCircle << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (randomX, randomY, resultCountsDevice);

    size_t* resultCountsHost = new size_t[BLOCKS_PER_GRID];
    cudaMemcpy(resultCountsHost, resultCountsDevice, BLOCKS_PER_GRID * sizeof(size_t), cudaMemcpyDeviceToHost);

    size_t result = 0;
    for (size_t i = 0; i < BLOCKS_PER_GRID; ++i)
    {
        result += resultCountsHost[i];
    }


    cudaFree(resultCountsDevice);
    delete[] resultCountsHost;

    return result;
}