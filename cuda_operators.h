#ifndef _CUDA_OPERATORS
#define _CUDA_OPERATORS


/// Class used to hold all the cuda functions that are required for simple vector, matrix math


template< typename STREAM > STREAM& operator<< ( STREAM& stm, const float4 f ) { return stm << f.x << " " << f.y << " " << f.z << " " << f.w; }
template< typename STREAM > STREAM& operator<< ( STREAM& stm, const float3 f ) { return stm << f.x << " " << f.y << " " << f.z; }

__host__ __device__ int toIndex(const int &x, const int &y, const size_t &width)
{
	return x+y*width;
}






#endif //_CUDA_OPERATORS
