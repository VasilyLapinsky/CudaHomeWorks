#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <iostream>

const size_t WIDTH = 32;
const size_t HEIGHT = WIDTH;

//const size_t BLOCKS_WIDTH = std::min(size_t(32), (WIDTH*HEIGHT + WIDTH - 1) / WIDTH);
//const size_t BLOCKS_HEIGHT = BLOCKS_WIDTH;

__device__ float GenerateUniform(curandState_t *state)
{
    return curand_uniform(state);
}


__global__ void CalculatePointsIntheCircle(unsigned int* result)
{
    curandState_t state;
    unsigned long long seed = (threadIdx.x + blockDim.x * blockIdx.x) * threadIdx.y + blockDim.y * blockIdx.y;
    curand_init(seed, 0, 0, &state);

    float x = GenerateUniform(&state);
    float y = GenerateUniform(&state);

    if (x * x + y * y <= 1.f)
    {
        atomicAdd(result, 1u);
    }
}

int main()
{
    // initialize clocks 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    unsigned int countedPoints = 0;
    unsigned int *countedPointsDevice;
    cudaMalloc((void**)&countedPointsDevice, sizeof(unsigned int));
    cudaMemcpy(countedPointsDevice, &countedPoints, sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 blocks(1, 1, 1);
    dim3 threads(HEIGHT, WIDTH, 1);
    CalculatePointsIntheCircle<<<blocks, threads>>>(countedPointsDevice);

    cudaMemcpy(&countedPoints, countedPointsDevice, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // calculates pi
    float pi = (4.f * static_cast<float>(countedPoints)) / static_cast<float>(HEIGHT*WIDTH);
    std::cout << "Result pi: " << pi << '\n';

    // calculate execution time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float   elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed time: " << elapsedTime << " ms\n";
    return 0;
}