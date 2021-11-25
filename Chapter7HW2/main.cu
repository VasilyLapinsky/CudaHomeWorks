#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <exception>
#include <string>
#include <chrono>
#include <type_traits>

const size_t NUM_THREADS = 512;

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

template<typename T>
__global__ void fill(T* matrix, int size, T val) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
		matrix[i] = val;
	}
}

__global__ void VectorMult(float* left, float* right, float* result, int size)
{
	extern __shared__ float resultcache[];

	const size_t cacheIndex = threadIdx.x;
	resultcache[cacheIndex] = 0;

	int tid = threadIdx.x;
	while (tid < size)
	{
		resultcache[cacheIndex] += left[tid] * right[tid];
		tid += gridDim.x * blockDim.x;
	}

	__syncthreads();

	// reduction
	size_t i = blockDim.x / 2;

	while (i != 0) {
		if (cacheIndex < i)
		{
			resultcache[cacheIndex] += resultcache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
	{
		*result = resultcache[0];
	}
}

int main()
{
	try
	{
		const size_t size = 30000;

		const auto startCalculation = std::chrono::system_clock::now();

		float* vector;
		HandleCudaStatus(cudaMalloc((void**)&vector, size * sizeof(float)));
		fill<<<1, NUM_THREADS >>>(vector, size, 5.f);
		HandleCudaStatus(cudaGetLastError());

		float* result;
		HandleCudaStatus(cudaMalloc((void**)&result, sizeof(float)));
		VectorMult<<<1, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(vector, vector, result, size);
		HandleCudaStatus(cudaGetLastError());

		float resultCpu;
		HandleCudaStatus(cudaMemcpy((void*)&resultCpu, result, sizeof(float), cudaMemcpyDeviceToHost));

		HandleCudaStatus(cudaFree(vector));
		HandleCudaStatus(cudaFree(result));

		std::cout << "result: " << sqrt(resultCpu) << '\n';
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startCalculation);
		std::cout << "Duration: " << duration.count() << " milliseconds\n";
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}
