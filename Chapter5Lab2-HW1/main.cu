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

__global__ void VectorMultFloat(float* left, float* right, float* result, int size)
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

__global__ void VectorMultDouble(double* left, double* right, double* result, int size)
{
	extern __shared__ double cache[];

	const size_t cacheIndex = threadIdx.x;
	cache[cacheIndex] = 0;

	int tid = threadIdx.x;
	while (tid < size)
	{
		cache[cacheIndex] += left[tid] * right[tid];
		tid += gridDim.x * blockDim.x;
	}

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
		*result = cache[0];
	}
}

std::chrono::system_clock::duration DoAnalysisFloat(const int size)
{
	const auto startCalculation = std::chrono::system_clock::now();

	float* left;
	HandleCudaStatus(cudaMalloc((void**)&left, size * sizeof(float)));
	fill<<<1, NUM_THREADS>>>(left, size, 1.f);
	HandleCudaStatus(cudaGetLastError());

	float* right;
	HandleCudaStatus(cudaMalloc((void**)&right, size * sizeof(float)));
	fill<<<1, NUM_THREADS>>>(right, size, 1.f);
	HandleCudaStatus(cudaGetLastError());

	float* result;
	HandleCudaStatus(cudaMalloc((void**)&result, sizeof(float)));
	VectorMultFloat<<<1, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(left, right, result, size);
	HandleCudaStatus(cudaGetLastError());

	HandleCudaStatus(cudaFree(left));
	HandleCudaStatus(cudaFree(right));
	HandleCudaStatus(cudaFree(result));

	return std::chrono::system_clock::now() - startCalculation;
}

std::chrono::system_clock::duration DoAnalysisDouble(const int size)
{
	const auto startCalculation = std::chrono::system_clock::now();

	double* left;
	HandleCudaStatus(cudaMalloc((void**)&left, size * sizeof(double)));
	fill<< <1, NUM_THREADS >> >(left, size, 1.);
	HandleCudaStatus(cudaGetLastError());

	double* right;
	HandleCudaStatus(cudaMalloc((void**)&right, size * sizeof(double)));
	fill << <1, NUM_THREADS >> > (right, size, 1.);
	HandleCudaStatus(cudaGetLastError());

	double* result;
	HandleCudaStatus(cudaMalloc((void**)&result, sizeof(double)));
	VectorMultDouble << <1, NUM_THREADS, NUM_THREADS * sizeof(double) >> > (left, right, result, size);
	HandleCudaStatus(cudaGetLastError());

	HandleCudaStatus(cudaFree(left));
	HandleCudaStatus(cudaFree(right));
	HandleCudaStatus(cudaFree(result));

	return std::chrono::system_clock::now() - startCalculation;
}

int main()
{
	try
	{
		const int SIZE = 1000000;
		const int NUM_REPS = 10000;
		std::chrono::system_clock::duration floatDuration = std::chrono::system_clock::duration::zero();
		for (int i = 0; i < NUM_REPS; ++i)
		{
			floatDuration += DoAnalysisFloat(SIZE);
		}
		std::cout << "Float Calculation time: " 
				<< std::chrono::duration_cast<std::chrono::milliseconds>(floatDuration / NUM_REPS).count() 
				<< " miliseconds\n";

		std::chrono::system_clock::duration doubleDuration = std::chrono::system_clock::duration::zero();
		for (int i = 0; i < NUM_REPS; ++i)
		{
			doubleDuration += DoAnalysisDouble(SIZE);
		}
		std::cout << "Double Calculation time: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(doubleDuration / NUM_REPS).count()
			<< " miliseconds\n";
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}
