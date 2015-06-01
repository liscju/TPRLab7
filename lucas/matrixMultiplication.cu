// Matrix multiplication by parts
// Elements stored in row-major order

using namespace std;
#include <stdio.h>
#include <iostream>
#include <fstream>

#include <cuda.h>
#include "helper_functions.h"

#define BLOCK_SIZE 16
#define ITER_COUNT 1000

typedef struct {
	int width;
	int height;
	float *elements;
} Matrix;

// Forward declaration of matrix mult
__global__ void MatMulKernel (const Matrix, const Matrix, Matrix);

// Host code
void MatMulGPU(const Matrix A, const Matrix B, Matrix C) {
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
    dim3 dimBlock(256); // threads per block?
    dim3 dimGrid(256); // number of blocks?

	// TIMER START
	StopWatchInterface *timer=NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	for(int i=0; i<ITER_COUNT; ++i) {
		MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	}

	// TIMER STOP
	sdkStopTimer(&timer);
	float time = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

	printf("GPU Elapsed time: %f\n", time/ITER_COUNT);
	
	// copy C to host
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	
	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

//matrix multiplication kernel
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {	
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > A.height || col > B.width) return;
	for (int e = 0; e < A.width; ++e) {
		Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
	}
	C.elements[row * C.width + col] = Cvalue;
}

void MatMulCPU(Matrix A, Matrix B, Matrix C) {
	// TIMER START
	StopWatchInterface *timer=NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	for(int i=0; i<ITER_COUNT; ++i) {		
		for(int row=0; row<A.height; row++) {
			for(int col=0; col<B.width; col++) {
				float Cvalue = 0;
				for (int e = 0; e < A.width; ++e) {
					Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
				}
				C.elements[row * C.width + col] = Cvalue;
			}
		}
	}

	// TIMER STOP
	sdkStopTimer(&timer);
	float time = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

	printf("CPU Elapsed time: %f\n", time/ITER_COUNT);
}

int main(int argc, char * const argv[]) {	
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
	A_input.open("A.txt");
	B_input.open("B.txt");
	
	float a, b;
	A_input >> a;	
	B_input >> b;	
	int i = 0;
	while (!A_input.eof()) {
		A.elements[i] = a;
		B.elements[i] = b;
		A_input >> a;	
		B_input >> b;	
		i += 1;
	}
	A_input.close();
	B_input.close();

	// TIMER START
	StopWatchInterface *timer=NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	MatMulGPU(A, B, C);

	// TIMER STOP
	sdkStopTimer(&timer);
	float gpuTime = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

	
	// TIMER START
	timer=NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	MatMulCPU(A, B, C);

	// TIMER STOP
	sdkStopTimer(&timer);
	float cpuTime = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);


	std::ofstream C_output;
	C_output.open("C.txt");
	for (int i=0; i<Width; i++) {
		for (int j=0; j<Width; j++) {
			C_output<<C.elements[i*Width+j]<<"\t";
		}
		C_output<<endl;
	}
	C_output.close();

	printf("GPU Elapsed total time: %f\n", gpuTime);
	printf("CPU Elapsed total time: %f\n", cpuTime);

	delete A.elements;
	delete B.elements;
	delete C.elements;

	cudaDeviceReset();
	return 0;
}
	
