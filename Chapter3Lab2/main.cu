
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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


__global__ void find_pi_number_using_integral(float* result)
{
	extern __shared__ float cache[];

	const float h = 1 / (float)(blockDim.x);
	const float x = h * threadIdx.x;
	cache[threadIdx.x] = sqrtf(1 - x * x) * h;

	__syncthreads();

	// reduction
	const size_t cacheIndex = threadIdx.x;
	size_t i = blockDim.x / 2;

	while (i != 0) {
		if (cacheIndex < i)
		{
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == i)
	{
		*result = 4.f * cache[0];
	}
}

int main()
{
	try
	{
		const size_t NUM_THREADS = 256;
		float *cudaPi;
		HandleCudaStatus(cudaMalloc((void**)&cudaPi, sizeof(float)));

		find_pi_number_using_integral<<<1, NUM_THREADS, NUM_THREADS  * sizeof(float)>>>(cudaPi);
		HandleCudaStatus(cudaGetLastError());

		float pi;
		HandleCudaStatus(cudaMemcpy(&pi, cudaPi, sizeof(float), cudaMemcpyDeviceToHost));

		std::cout << "Pi: " << pi << std::endl;
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}
