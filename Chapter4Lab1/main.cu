#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include <stdio.h>
#include <iostream>
#include <exception>
#include <string>

void HandleCudaStatus(cudaError status) {
	switch (status)
	{
		case cudaSuccess: break;
		case cudaErrorMemoryAllocation: throw std::exception("Error in memory allocation");
		case cudaErrorInvalidValue: throw std::exception("Invalid argument value");
		case cudaErrorInvalidDevicePointer: throw std::exception("Invalid device pointer");
		case cudaErrorInvalidMemcpyDirection: throw std::exception("Invalid copy dirrection");
		case cudaErrorInitializationError: throw std::exception("Error during initialization");
		case cudaErrorPriorLaunchFailure: throw std::exception("Error in previous launch");
		case cudaErrorInvalidResourceHandle: throw std::exception("Invalid resource handler");
		default: throw std::exception(("Unrecognized cuda status: " + std::to_string(static_cast<int>(status))).c_str());
	}
}


__device__ bool IsInCircle(curandState_t* state)
{
	float x = curand_uniform(state);
	float y = curand_uniform(state);
	return x * x + y * y <= 1.0f;
}


__global__ void CountPointsIntheCircle(unsigned int* result)
{

	//init curand
	curandState_t state;
	unsigned long long seed = (threadIdx.x + blockDim.x * blockIdx.x) + (threadIdx.y + blockDim.y * blockIdx.y) % 1000;
	curand_init(seed, 0, 0, &state);

	if (IsInCircle(&state))
	{
		atomicAdd(*result, 1);
	}
}

int main()
{
	try
	{
		const size_t NUM_THREADS = 512;
		unsigned int *cudaCounter;
		HandleCudaStatus(cudaMalloc((void**)&cudaCounter, sizeof(unsigned int)));

		CountPointsIntheCircle<<<1, NUM_THREADS>>>(cudaCounter);
		HandleCudaStatus(cudaGetLastError());

		unsigned int counter;
		HandleCudaStatus(cudaMemcpy(&counter, cudaCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

		std::cout << "Pi: " << static_cast<float>(counter) / static_cast<float>(NUM_THREADS) << std::endl;
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}
