#!/bin/sh
nvcc matrixMultiplication.cu -o matMul.out -gencode arch=compute_30,code=sm_30
