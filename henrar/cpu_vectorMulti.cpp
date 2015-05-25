#include <iostream>
#include <fstream>
#include "helper_functions"

class Matrix
{
public:	
	int width;
	int height;
	float *elements;
};


void prepareMatrices()
{
	std::fstream A_input;
	std::fstream B_input;

	const int Width = 16;

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
}

int main()
{
	prepareMatrices();
	return 0;
}
