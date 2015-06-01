// Matrix multiplication by parts
// Elements stored in row-major order

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include "helper_functions.h"
#include "helper_cuda.h"

#define BLOCK_SIZE 16

class Matrix
{
public:	
	int width;
	int height;
	float *elements;
};

// Forward declaration of matrix mult
__global__ void MatMulKernel (const Matrix, const Matrix, Matrix);

// Host code
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	StopWatchInterface *timer = NULL;
	float elapsedTime = 0.0f;
	cudaEvent_t start, stop;
	// Load matrices A and B to device memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc((void**) &d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	
	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc((void**) &d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	
	// allocate C in device
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = d_C.width * d_C.height * sizeof(float);
	cudaMalloc((void**) &d_C.elements, size);
	
	// call kernel
        dim3 dimBlock(128); // threads per block?
        dim3 dimGrid(128); // number of blocks?


	sdkCreateTimer(&timer);
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));


	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&timer);
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	std::cout << "Time: " << elapsedTime << std::endl;
	// copy C to host
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

//matrix multiplication kernel
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{	
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > A.height || col > B.width)
	{
		return;
	}
	for (int e = 0; e < A.width; ++e)
	{
		Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
	}
	C.elements[row * C.width + col] = Cvalue;
}

int main(int argc, char * const argv[])
{	
	int Width = 16;
	
	Matrix A;
	Matrix B;
	Matrix C;
	
	A.width = Width;
	B.width = Width;
	C.width = Width;
	
	A.height = Width;
	B.height = Width;
	C.height = Width;
	
	A.elements = new float[Width*Width];
	B.elements = new float[Width*Width];
	C.elements = new float[Width*Width];
	
	//fill matrices
	std::ifstream A_input;
	std::ifstream B_input;
	A_input.open("input/A.txt");
	B_input.open("input/B.txt");
	
	float a, b;
	A_input >> a;	
	B_input >> b;	
	int i = 0;
	while (!A_input.eof())
	{	A.elements[i] = a;
		B.elements[i] = b;
		A_input >> a;	
		B_input >> b;	
		i += 1;
	}
	A_input.close();
	B_input.close();

	MatMul(A, B, C);
	std::ofstream C_output;
	C_output.open("output/gpu_results.txt");
	for (int i = 0; i < Width; i++)
	{	for (int j = 0; j < Width; j++)
		{
			C_output << C.elements[i*Width+j]<<"\t";
		}
		C_output << std::endl;
	}
	
	C_output.close();
	cudaDeviceReset();
	return 0;
}
	
