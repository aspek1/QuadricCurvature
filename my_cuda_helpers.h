#ifndef _MY_CUDA_HELP
#define _MY_CUDA_HELP

#undef PRINT_DEBUG

#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


template <class T>
T* cudaArray(size_t length, size_t bytes, bool zero_array = true);
template <class T>
inline void printCudaArray(const T* const d_arr, size_t size, size_t bytes, int npl);
inline void cudaCheck(int errCode, const char* str);

/// Overloads to allow correct printing of object in ostream



inline void cudaCheck(int errCode, const char* str = "")
{
#ifdef PRINT_DEBUG
	std::cout << "Checking Cuda Errors: ";
#endif
	if(errCode != CUDA_SUCCESS)
	{
		std::cerr << str <<  " >> CUDA ERROR: " << cudaGetErrorString((cudaError_t)errCode) << " - exiting" << std::endl;
		exit(errCode);
	}
#ifdef PRINT_DEBUG
	std::cout << "No Error(s)" << std::endl;
#endif
}
// creates a CUDA array of any type, and optionally zeros all values
template < class T >
T* cudaArray(size_t length,
		size_t bytes,
		bool zero_array = true)
{

#ifdef PRINT_DEBUG
	std::cout << "Creating Cuda Array: " << length << ", " << bytes << ", " << zero_array << std::endl;
#endif
	T* d_array;
	cudaCheck(cudaMalloc((void**) &d_array, bytes*length));
	if(zero_array)
		cudaCheck(cudaMemset(d_array, 0, bytes*length));
	return d_array;
}


// Prints any CUDA array by first copying it back to the CPU
template < typename T >
inline void printCudaArray(const T* const d_arr,
		size_t length,
		size_t bytes,
		int npl = 100000)
{

#ifdef PRINT_DEBUG
	std::cout << "Printing Cuda Array: " << length << ", " << bytes << ", " << npl << std::endl;
#endif

	T h_arr[length];

	cuda_check(cudaMemcpy(h_arr, d_arr, bytes*length, cudaMemcpyDeviceToHost));

	for(size_t i = 0; i<length; ++i){
		if(i > 0 && (i%npl) ==  0) std::cout << std::endl;
		std::cout << (T)h_arr[i] << " ";
	}
	std::cout << std::endl;
}



#endif //_MY_CUDA_HELP
