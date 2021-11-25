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

__global__ void fill(double* matrix, int size, double val) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
		matrix[i] = val;
	}
}

__global__ void matrix_mult(double* left, double* right, double* result, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < k && row < m)
	{
		double sum = 0;
		for (int i = 0; i < n; i++)
		{
			sum += left[row * n + i] * right[i * k + col];
		}
		result[row * k + col] = sum;
	}
}

__global__ void fill(double* a, double* b, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
		a[i] = b[i];
	}
}

void CheckCommutation()
{
	const int N = 10;
	const int DATA_SIZE = N * N;
	const int NUM_THREADS = 512;

	double* left;
	HandleCudaStatus(cudaMalloc((void**)&left, DATA_SIZE * sizeof(double)));
	fill<<<1, NUM_THREADS>>>(left, DATA_SIZE, 1.);

	double* right;
	HandleCudaStatus(cudaMalloc((void**)&right, DATA_SIZE * sizeof(double)));
	fill<<<1, NUM_THREADS>>>(right, DATA_SIZE, 5.);

	double* leftToRight;
	HandleCudaStatus(cudaMalloc((void**)&leftToRight, DATA_SIZE * sizeof(double)));

	double* rightToLeft;
	HandleCudaStatus(cudaMalloc((void**)&rightToLeft, DATA_SIZE * sizeof(double)));

	const int BLOCK_SIZE = 16;
	int gridRows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int gridCols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(gridCols, gridRows);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	matrix_mult<<<dimGrid, dimBlock>>>(left, right, leftToRight, N, N, N);
	HandleCudaStatus(cudaGetLastError());
	matrix_mult<<<dimGrid, dimBlock>>>(right, left, rightToLeft, N, N, N);
	HandleCudaStatus(cudaGetLastError());

	double* leftToRightCpu = new double[DATA_SIZE];
	HandleCudaStatus(cudaMemcpy((void*)leftToRightCpu, (void*)leftToRight, DATA_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
	double* rightToLeftCpu = new double[DATA_SIZE];
	HandleCudaStatus(cudaMemcpy((void*)rightToLeftCpu, (void*)rightToLeft, DATA_SIZE * sizeof(double), cudaMemcpyDeviceToHost));

	for (int i = 0; i < DATA_SIZE; ++i)
	{
		if (leftToRightCpu[i] != rightToLeftCpu[i])
		{
			std::cout << "Matrix are not commutative\n";
			return;
		}
	}
	std::cout << "Matrix are commutative\n";
}

int main()
{
	try
	{
		CheckCommutation();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}
