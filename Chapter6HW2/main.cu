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

__global__ void add(float* left, float* right, float* result, int size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < size) {
		result[tid] = left[tid] + right[tid];
		tid += blockDim.x * gridDim.x;
	}
}

float* ToGpuMatrix(float** matrixCpu, int m, int n)
{
	float* matrixGpu;
	HandleCudaStatus(cudaMalloc((void**)&matrixGpu, m * n * sizeof(float)));
	for (int i = 0; i < m; ++i)
	{
		HandleCudaStatus(cudaMemcpy((void*)(matrixGpu + i*n), (void*)matrixCpu[i], n * sizeof(float), cudaMemcpyHostToDevice));
	}

	return matrixGpu;
}

float** Add(float** a, float** b, int m, int n)
{
	float* gpuA = ToGpuMatrix(a, m, n);
	float* gpuB = ToGpuMatrix(b, m, n);

	const int size = m * n;
	float* resultGpu;
	HandleCudaStatus(cudaMalloc((void**)&resultGpu, size * sizeof(float)));

	const int MAX_THREADS = 256;
	dim3 grids((size + MAX_THREADS - 1) / MAX_THREADS);
	dim3 threads(MAX_THREADS);
	add<<<grids, threads>>>(gpuA, gpuB, resultGpu, size);

	float** resultCpu = new float*[m];
	for (int i = 0; i < m; ++i)
	{
		resultCpu[i] = new float[n];
		HandleCudaStatus(cudaMemcpy((void*)resultCpu[i], (void*)(resultGpu + i * n), n * sizeof(float), cudaMemcpyDeviceToHost));
	}

	cudaFree(gpuA);
	cudaFree(gpuB);
	cudaFree(resultGpu);

	return resultCpu;
}

float** CreateRandom(int m, int n)
{
	float** matrix = new float* [m];
	for (int i = 0; i < m; ++i)
	{
		matrix[i] = new float[n];
		for (int j = 0; j < n; ++j)
		{
			matrix[i][j] = std::rand() % 10;
		}
	}

	return matrix;
}

void print(float** matrix, int m, int n)
{
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			std::cout << matrix[i][j] << ' ';
		}
		std::cout << '\n';
	}
}

int main()
{
	try
	{
		const int m = 10;
		const int n = 5;
		auto a = CreateRandom(m, n);
		auto b = CreateRandom(m, n);
		auto result = Add(a, b, m, n);

		std::cout << "a: \n";
		print(a, m, n);
		std::cout << "b: \n";
		print(b, m, n);
		std::cout << "result: \n";
		print(result, m, n);
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}
