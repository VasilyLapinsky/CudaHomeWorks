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
		case cudaErrorIllegalAddress: throw std::exception("Illegal address");
		default: throw std::exception(("Unrecognized cuda status: " + std::to_string(static_cast<int>(status))).c_str());
	}
}

typedef float (*method_t)(float, float);

__device__ float function(float x)
{
	return sin(x) * x + exp(x);
}

__device__ float squares(float x, float step)
{
	return function(x + step / 2.f) * step;
}

__device__ float trapezoids(float x, float step)
{
	return step * (function(x) + function(x+step)) / 2.f;
}

__device__ float simpson(float x, float step)
{
	return step * (function(x) + 4.f * function(x + step / 2.f) + function(x + step)) / 6.f;
}

__global__ void Integrate(float begin, float end, size_t numPoints, float step, float* result, method_t* method)
{
	extern __shared__ float resultcache[];

	const size_t cacheIndex = threadIdx.x;
	resultcache[cacheIndex] = 0;

	int tid = threadIdx.x;
	float x;
	while (tid < numPoints)
	{
		x = step * tid;
		resultcache[cacheIndex] += (*method)(x, step);
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

__device__ method_t squares_pointer = squares;
__device__ method_t trapezoids_pointer = trapezoids;
__device__ method_t simpson_pointer = simpson;

int main()
{
	try
	{
		const size_t NUMBER_OF_POINTS = 1000000;
		const float BEGIN = 2;
		const float END = 10;
		const float STEP = (END - BEGIN) / static_cast<float>(NUMBER_OF_POINTS);
		
		float* result;
		HandleCudaStatus(cudaMalloc((void**)&result, sizeof(float)));

		method_t* hostPointer = new method_t;
		method_t* devicePointer;
		HandleCudaStatus(cudaMalloc((void**)&devicePointer, sizeof(method_t)));

		cudaMemcpyFromSymbol(hostPointer, squares_pointer, sizeof(method_t));
		cudaMemcpy(devicePointer, hostPointer, sizeof(method_t), cudaMemcpyHostToDevice);

		auto startCalculation = std::chrono::system_clock::now();
		Integrate <<<1, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(BEGIN, END, NUMBER_OF_POINTS, STEP, result, devicePointer);
		HandleCudaStatus(cudaGetLastError());
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startCalculation);
		float resultSquares;
		HandleCudaStatus(cudaMemcpy((void*)&resultSquares, result, sizeof(float), cudaMemcpyDeviceToHost));
		std::cout << "result squares: " << resultSquares << '\n';
		std::cout << "Duration squares: " << duration.count() << " milliseconds\n";

		cudaMemcpyFromSymbol(hostPointer, trapezoids_pointer, sizeof(method_t));
		cudaMemcpy(devicePointer, hostPointer, sizeof(method_t), cudaMemcpyHostToDevice);

		startCalculation = std::chrono::system_clock::now();
		Integrate<<<1, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(BEGIN, END, NUMBER_OF_POINTS, STEP, result, devicePointer);
		HandleCudaStatus(cudaGetLastError());
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startCalculation);
		float resultTrapezoid;
		HandleCudaStatus(cudaMemcpy((void*)&resultTrapezoid, result, sizeof(float), cudaMemcpyDeviceToHost));
		std::cout << "result trapezoids: " << resultTrapezoid << '\n';
		std::cout << "Duration trapezoids: " << duration.count() << " milliseconds\n";

		cudaMemcpyFromSymbol(hostPointer, simpson_pointer, sizeof(method_t));
		cudaMemcpy(devicePointer, hostPointer, sizeof(method_t), cudaMemcpyHostToDevice);

		startCalculation = std::chrono::system_clock::now();
		Integrate<<<1, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(BEGIN, END, NUMBER_OF_POINTS, STEP, result, devicePointer);
		HandleCudaStatus(cudaGetLastError());
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startCalculation);
		float resultSimpson;
		HandleCudaStatus(cudaMemcpy((void*)&resultSimpson, result, sizeof(float), cudaMemcpyDeviceToHost));
		std::cout << "result simpson: " << resultSimpson << '\n';
		std::cout << "Duration simpson: " << duration.count() << " milliseconds\n";
		HandleCudaStatus(cudaFree(result));


		std::cout << "Squares - Trapezoid = " << resultSquares - resultTrapezoid << '\n';
		std::cout << "Squares - Simpson = " << resultSquares - resultSimpson << '\n';
		std::cout << "Trapezoid - Simpson = " << resultTrapezoid - resultSimpson << '\n';
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}
