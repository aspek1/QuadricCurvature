#ifndef CUDA_VEC_CUH
#define CUDA_VEC_CUH

#include "vector_types.h"
#include <cuda_runtime.h>

// Vector and matrix functions for use with the CUDA built in data types
// Based on Vec.h, ported to CUDA by Stefan Auer

/********************************************************************

TUM3D Vector/Matrix/Utils Header

copyright 2008- 2009

Joachim Georgii, georgii@tum.de
Roland Fraedrich, fraedrich@tum.de
Christian Dick, dick@tum.de
Stefan Auer, auer@in.tum.de

VERSION      1.00
DATE         29.05.2009
LAST_CHANGE  SA

*******************************************************************/

#define CUDA_VEC_ASSERT(expression) { if(! (expression) ) *(int*)0=0; }

#ifdef __CUDACC__
#define CUDA_VEC_HOSTDEVICE inline __host__ __device__
#else
#define CUDA_VEC_HOSTDEVICE inline
#endif


#ifndef __CUDACC__
#include <math.h>

// float functions, host only

CUDA_VEC_HOSTDEVICE float fminf(float a, float b)
{
  return a < b ? a : b;
}

CUDA_VEC_HOSTDEVICE float fmaxf(float a, float b)
{
  return a > b ? a : b;
}

#undef max
CUDA_VEC_HOSTDEVICE int max(int a, int b)
{
  return a > b ? a : b;
}

#undef min
CUDA_VEC_HOSTDEVICE int min(int a, int b)
{
  return a < b ? a : b;
}

CUDA_VEC_HOSTDEVICE float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}
#endif

// float functions
////////////////////////////////////////////////////////////////////////////////

// lerp
CUDA_VEC_HOSTDEVICE float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

// clamp
CUDA_VEC_HOSTDEVICE float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

// int functions
////////////////////////////////////////////////////////////////////////////////

// clamp
CUDA_VEC_HOSTDEVICE int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}

// uint functions
////////////////////////////////////////////////////////////////////////////////

// clamp
CUDA_VEC_HOSTDEVICE unsigned int clamp(unsigned int f, unsigned int a, unsigned int b)
{
    return max(a, min(f, b));
}

// float2 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
CUDA_VEC_HOSTDEVICE float2 make_float2(float const * a)
{
    return make_float2(a[0], a[1]);
}
CUDA_VEC_HOSTDEVICE float2 make_float2(float s)
{
    return make_float2(s, s);
}
CUDA_VEC_HOSTDEVICE float2 make_float2(float3 const & a)
{
    return make_float2(a.x, a.y);  // discards z
}
CUDA_VEC_HOSTDEVICE float2 make_float2(float4 const & a)
{
    return make_float2(a.x, a.y);  // discards z, w
}
CUDA_VEC_HOSTDEVICE float2 make_float2(int2 const & a)
{
    return make_float2(float(a.x), float(a.y));
}
CUDA_VEC_HOSTDEVICE float2 make_float2(uint2 const & a)
{
    return make_float2(float(a.x), float(a.y));
}

// negate
CUDA_VEC_HOSTDEVICE float2 operator-(float2 const & a)
{
    return make_float2(-a.x, -a.y);
}

// min
CUDA_VEC_HOSTDEVICE float2 fminf(float2 const & a, float2 const & b)
{
	return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}

// max
CUDA_VEC_HOSTDEVICE float2 fmaxf(float2 const & a, float2 const & b)
{
	return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}

// addition
CUDA_VEC_HOSTDEVICE float2 operator+(float2 const & a, float2 const & b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
CUDA_VEC_HOSTDEVICE float2 operator+(float2 const & a, float b)
{
    return make_float2(a.x + b, a.y + b);
}
CUDA_VEC_HOSTDEVICE float2 operator+(float a, float2 const & b)
{
    return make_float2(a + b.x, a + b.y);
}
CUDA_VEC_HOSTDEVICE void operator+=(float2 & a, float2 const & b)
{
    a.x += b.x; a.y += b.y;
}
CUDA_VEC_HOSTDEVICE void operator+=(float2 & a, float b)
{
    a.x += b; a.y += b;
}

// subtract
CUDA_VEC_HOSTDEVICE float2 operator-(float2 const & a, float2 const & b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
CUDA_VEC_HOSTDEVICE float2 operator-(float2 const & a, float b)
{
    return make_float2(a.x - b, a.y - b);
}
CUDA_VEC_HOSTDEVICE float2 operator-(float a, float2 const & b)
{
    return make_float2(a - b.x, a - b.y);
}
CUDA_VEC_HOSTDEVICE void operator-=(float2 & a, float2 const & b)
{
    a.x -= b.x; a.y -= b.y;
}
CUDA_VEC_HOSTDEVICE void operator-=(float2 & a, float b)
{
    a.x -= b; a.y -= b;
}

// multiply
CUDA_VEC_HOSTDEVICE float2 operator*(float2 const & a, float2 const & b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
CUDA_VEC_HOSTDEVICE float2 operator*(float2 const & a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
CUDA_VEC_HOSTDEVICE float2 operator*(float a, float2 const & b)
{
    return make_float2(a * b.x, a * b.y);
}
CUDA_VEC_HOSTDEVICE void operator*=(float2 & a, float2 const & b)
{
    a.x *= b.x; a.y *= b.y;
}
CUDA_VEC_HOSTDEVICE void operator*=(float2 & a, float b)
{
    a.x *= b; a.y *= b;
}

// divide
CUDA_VEC_HOSTDEVICE float2 operator/(float2 const & a, float2 const & b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
CUDA_VEC_HOSTDEVICE float2 operator/(float2 const & a, float b)
{
    float inv = 1.0f / b;
    return a * inv;
}
CUDA_VEC_HOSTDEVICE float2 operator/(float a, float2 const & b)
{
    return make_float2(a / b.x, a / b.y);
}
CUDA_VEC_HOSTDEVICE void operator/=(float2 & a,  float2 const & b)
{
    a.x /= b.x; a.y /= b.y;
}
CUDA_VEC_HOSTDEVICE void operator/=(float2 & a, float b)
{
    float inv = 1.0f / b;
    a *= inv;
}

// lerp
CUDA_VEC_HOSTDEVICE float2 lerp(float2 const & a, float2 const & b, float t)
{
    return a + t*(b-a);
}

// clamp
CUDA_VEC_HOSTDEVICE float2 clamp(float2 const & v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

CUDA_VEC_HOSTDEVICE float2 clamp(float2 const & v, float2 const & a, float2 const & b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

// dot product
CUDA_VEC_HOSTDEVICE float dot(float2 const & a, float2 const & b)
{
    return a.x * b.x + a.y * b.y;
}

// length
CUDA_VEC_HOSTDEVICE float length(float2 const & a)
{
    return sqrtf(dot(a, a));
}

// normalize
CUDA_VEC_HOSTDEVICE float2 normalize(float2 const & a)
{
    float invLen = rsqrtf(dot(a, a));
    return a * invLen;
}

// floor
CUDA_VEC_HOSTDEVICE float2 floor(float2 const & a)
{
    return make_float2(floor(a.x), floor(a.y));
}

// ceil
CUDA_VEC_HOSTDEVICE float2 ceil(float2 const & a)
{
    return make_float2(ceil(a.x), ceil(a.y));
}

// reflect
CUDA_VEC_HOSTDEVICE float2 reflect(float2 const & i, float2 const & n)
{
	return i - 2.0f * n * dot(n,i);
}

// absolute value
CUDA_VEC_HOSTDEVICE float2 fabs(float2 const & a)
{
	return make_float2(fabs(a.x), fabs(a.y));
}

// log
CUDA_VEC_HOSTDEVICE float2 log(float2 const & a)
{
	return make_float2(log(a.x), log(a.y));
}

// norm square
CUDA_VEC_HOSTDEVICE float norm_square(float2 const & a)
{
	return a.x * a.x + a.y * a.y;
}

// norm
CUDA_VEC_HOSTDEVICE float norm(float2 const & a)
{
	return sqrt(norm_square(a));
}

// float3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
CUDA_VEC_HOSTDEVICE float3 make_float3(const float * a)
{
    return make_float3(a[0], a[1], a[2]);
}
CUDA_VEC_HOSTDEVICE float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
CUDA_VEC_HOSTDEVICE float3 make_float3(float2 const & a)
{
    return make_float3(a.x, a.y, 0.0f);
}
CUDA_VEC_HOSTDEVICE float3 make_float3(float2 const & a, float z)
{
    return make_float3(a.x, a.y, z);
}
CUDA_VEC_HOSTDEVICE float3 make_float3(float4 const & a)
{
    return make_float3(a.x, a.y, a.z);  // discards w
}
CUDA_VEC_HOSTDEVICE float3 make_float3(int3 const & a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
CUDA_VEC_HOSTDEVICE float3 make_float3(int2 const & a, int z)
{
    return make_float3(float(a.x), float(a.y), float(z));
}
CUDA_VEC_HOSTDEVICE float3 make_float3(uint3 const & a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
CUDA_VEC_HOSTDEVICE float3 make_float3(uint2 const & a, unsigned int z)
{
    return make_float3(float(a.x), float(a.y), float(z));
}

// negate
CUDA_VEC_HOSTDEVICE float3 operator-(float3 const &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// min
CUDA_VEC_HOSTDEVICE float3 fminf(float3 const & a, float3 const & b)
{
	return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

// max
CUDA_VEC_HOSTDEVICE float3 fmaxf(float3 const & a, float3 const & b)
{
	return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

// addition
CUDA_VEC_HOSTDEVICE float3 operator+(float3 const & a, float3 const & b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
CUDA_VEC_HOSTDEVICE float3 operator+(float3 const & a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
CUDA_VEC_HOSTDEVICE float3 operator+(float a, float3 const & b)
{
    return make_float3(a + b.x, a + b.y, a + b.z);
}
CUDA_VEC_HOSTDEVICE void operator+=(float3 & a, float3 const & b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
CUDA_VEC_HOSTDEVICE void operator+=(float3 & a, float b)
{
    a.x += b; a.y += b; a.z += b;
}

// subtract
CUDA_VEC_HOSTDEVICE float3 operator-(float3 const & a, float3 const & b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
CUDA_VEC_HOSTDEVICE float3 operator-(float3 const & a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
CUDA_VEC_HOSTDEVICE float3 operator-(float a, float3 const & b)
{
    return make_float3(a - b.x, a - b.y, a - b.z);
}
CUDA_VEC_HOSTDEVICE void operator-=(float3 &a, float3 const & b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
CUDA_VEC_HOSTDEVICE void operator-=(float3 &a, float b)
{
    a.x -= b; a.y -= b; a.z -= b;
}

// multiply
CUDA_VEC_HOSTDEVICE float3 operator*(float3 const & a, float3 const & b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
CUDA_VEC_HOSTDEVICE float3 operator*(float3 const & a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
CUDA_VEC_HOSTDEVICE float3 operator*(float a, float3 const & b)
{
    return make_float3(a * b.x, a* b.y , a* b.z);
}
CUDA_VEC_HOSTDEVICE void operator*=(float3 &a, float3 const & b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}
CUDA_VEC_HOSTDEVICE void operator*=(float3 &a, float b)
{
    a.x *= b; a.y *= b; a.z *= b;
}

// divide
CUDA_VEC_HOSTDEVICE float3 operator/(float3 const & a, float3 const & b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
CUDA_VEC_HOSTDEVICE float3 operator/(float3 const & a, float b)
{
    float inv = 1.0f / b;
    return a * inv;
}
CUDA_VEC_HOSTDEVICE float3 operator/(float a, float3 const & b)
{
    return make_float3(a / b.x, a / b.y, a / b.z);
}
CUDA_VEC_HOSTDEVICE void operator/=(float3 &a, float3 const & b)
{
    a.x /= b.x; a.y /= b.y; a.z /= b.z;
}
CUDA_VEC_HOSTDEVICE void operator/=(float3 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
CUDA_VEC_HOSTDEVICE float3 lerp(float3 const & a, float3 const & b, float t)
{
    return a + t*(b-a);
}

// clamp
CUDA_VEC_HOSTDEVICE float3 clamp(float3 const & v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

CUDA_VEC_HOSTDEVICE float3 clamp(float3 const & v, float3 const & a, float3 const & b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// dot product
CUDA_VEC_HOSTDEVICE float dot(float3 const & a, float3 const & b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product
CUDA_VEC_HOSTDEVICE float3 cross(float3 const & a, float3 const & b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

// length
CUDA_VEC_HOSTDEVICE float length(float3 const & a)
{
    return sqrtf(dot(a, a));
}

// normalize
CUDA_VEC_HOSTDEVICE float3 normalize(float3 const & a)
{
    float invLen = rsqrtf(dot(a, a));
    return a * invLen;
}

// floor
CUDA_VEC_HOSTDEVICE float3 floor(float3 const & a)
{
    return make_float3(floor(a.x), floor(a.y), floor(a.z));
}

// ceil
CUDA_VEC_HOSTDEVICE float3 ceil(float3 const & a)
{
    return make_float3(ceil(a.x), ceil(a.y), ceil(a.z));
}

// reflect
CUDA_VEC_HOSTDEVICE float3 reflect(float3 const & i, float3 const & n)
{
	return i - 2.0f * n * dot(n,i);
}

// absolute value
CUDA_VEC_HOSTDEVICE float3 fabs(float3 const & a)
{
	return make_float3(fabs(a.x), fabs(a.y), fabs(a.z));
}

// log
CUDA_VEC_HOSTDEVICE float3 log(float3 const & a)
{
	return make_float3(log(a.x), log(a.y), log(a.z));
}

// norm square
CUDA_VEC_HOSTDEVICE float norm_square(float3 const & a)
{
	return a.x * a.x + a.y * a.y + a.z * a.z;
}

// norm
CUDA_VEC_HOSTDEVICE float norm(float3 const & a)
{
	return sqrt(norm_square(a));
}

// float4 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
CUDA_VEC_HOSTDEVICE float4 make_float4(float const * a)
{
    return make_float4(a[0], a[1], a[2], a[3]);
}
CUDA_VEC_HOSTDEVICE float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
CUDA_VEC_HOSTDEVICE float4 make_float4(float2 const & a)
{
    return make_float4(a.x, a.y, 0.0f, 0.0f);
}
CUDA_VEC_HOSTDEVICE float4 make_float4(float3 const & a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
CUDA_VEC_HOSTDEVICE float4 make_float4(float2 const & a, float z, float w)
{
    return make_float4(a.x, a.y, z, w);
}
CUDA_VEC_HOSTDEVICE float4 make_float4(float3 const & a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
CUDA_VEC_HOSTDEVICE float4 make_float4(int4 const & a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
CUDA_VEC_HOSTDEVICE float4 make_float4(int2 const & a, int z, int w)
{
    return make_float4(float(a.x), float(a.y), float(z), float(w));
}
CUDA_VEC_HOSTDEVICE float4 make_float4(int3 const & a, int w)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(w));
}
CUDA_VEC_HOSTDEVICE float4 make_float4(uint4 const & a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
CUDA_VEC_HOSTDEVICE float4 make_float4(uint2 const & a, unsigned int z, unsigned int w)
{
    return make_float4(float(a.x), float(a.y), float(z), float(w));
}
CUDA_VEC_HOSTDEVICE float4 make_float4(uint3 const & a, unsigned int w)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(w));
}

// negate
CUDA_VEC_HOSTDEVICE float4 operator-(float4 const & a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

// min
CUDA_VEC_HOSTDEVICE float4 fminf(float4 const & a, float4 const & b)
{
	return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

// max
CUDA_VEC_HOSTDEVICE float4 fmaxf(float4 const & a, float4 const & b)
{
	return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

// addition
CUDA_VEC_HOSTDEVICE float4 operator+(float4 const & a, float4 const & b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
CUDA_VEC_HOSTDEVICE float4 operator+(float4 const & a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
CUDA_VEC_HOSTDEVICE float4 operator+(float a, float4 const & b)
{
    return make_float4(a + b.x, a + b.y, a + b.z, a + b.w);
}
CUDA_VEC_HOSTDEVICE void operator+=(float4 &a, float4 const & b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
CUDA_VEC_HOSTDEVICE void operator+=(float4 &a, float b)
{
    a.x += b; a.y += b; a.z += b; a.w += b;
}

// subtract
CUDA_VEC_HOSTDEVICE float4 operator-(float4 const & a, float4 const & b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
CUDA_VEC_HOSTDEVICE float4 operator-(float4 const & a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
CUDA_VEC_HOSTDEVICE float4 operator-(float a, float4 const & b)
{
    return make_float4(a - b.x, a - b.y, a - b.z,  a - b.w);
}
CUDA_VEC_HOSTDEVICE void operator-=(float4 &a, float4 const & b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
CUDA_VEC_HOSTDEVICE void operator-=(float4 &a, float b)
{
    a.x -= b; a.y -= b; a.z -= b; a.w -= b;
}

// multiply
CUDA_VEC_HOSTDEVICE float4 operator*(float4 const & a, float4 const & b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
CUDA_VEC_HOSTDEVICE float4 operator*(float4 const & a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
CUDA_VEC_HOSTDEVICE float4 operator*(float a, float4 const & b)
{
    return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}
CUDA_VEC_HOSTDEVICE void operator*=(float4 &a, float4 const & b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}
CUDA_VEC_HOSTDEVICE void operator*=(float4 &a, float b)
{
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

// divide
CUDA_VEC_HOSTDEVICE float4 operator/(float4 const & a, float4 const & b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
CUDA_VEC_HOSTDEVICE float4 operator/(float4 const & a, float b)
{
    float inv = 1.0f / b;
    return a * inv;
}
CUDA_VEC_HOSTDEVICE float4 operator/(float a, float4 const & b)
{
    return make_float4(a / b.x, a / b.y, a / b.z, a / b.w);
}
CUDA_VEC_HOSTDEVICE void operator/=(float4 &a, float4 const & b)
{
    a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
}
CUDA_VEC_HOSTDEVICE void operator/=(float4 &a, float b)
{
    float inv = 1.0f / b;
    a *= inv;
}

// lerp
CUDA_VEC_HOSTDEVICE float4 lerp(float4 const & a, float4 const & b, float t)
{
    return a + t * (b - a);
}

// clamp
CUDA_VEC_HOSTDEVICE float4 clamp(float4 const & v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

CUDA_VEC_HOSTDEVICE float4 clamp(float4 const & v, float4 const & a, float4 const & b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

// dot product
CUDA_VEC_HOSTDEVICE float dot(float4 const & a, float4 const & b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// length
CUDA_VEC_HOSTDEVICE float length(float4 const & a)
{
    return sqrtf(dot(a, a));
}

// normalize
CUDA_VEC_HOSTDEVICE float4 normalize(float4 const & a)
{
    float invLen = rsqrtf(dot(a, a));
    return a * invLen;
}

// floor
CUDA_VEC_HOSTDEVICE float4 floor(float4 const & a)
{
    return make_float4(floor(a.x), floor(a.y), floor(a.z), floor(a.w));
}

// ceil
CUDA_VEC_HOSTDEVICE float4 ceil(float4 const & a)
{
    return make_float4(ceil(a.x), ceil(a.y), ceil(a.z), ceil(a.w));
}

// absolute value
CUDA_VEC_HOSTDEVICE float4 fabs(float4 const & a)
{
	return make_float4(fabs(a.x), fabs(a.y), fabs(a.z), fabs(a.w));
}

// log
CUDA_VEC_HOSTDEVICE float4 log(float4 const & a)
{
	return make_float4(log(a.x), log(a.y), log(a.z), log(a.w));
}

// norm square
CUDA_VEC_HOSTDEVICE float norm_square(float4 const & a)
{
	return a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
}

// norm
CUDA_VEC_HOSTDEVICE float norm(float4 const & a)
{
	return sqrt(norm_square(a));
}

CUDA_VEC_HOSTDEVICE float4 transform(float4 const & coord, const float4 * transform)
{
	float4 c = make_float4(dot(transform[0], coord), dot(transform[1], coord), dot(transform[2], coord), dot(transform[3], coord));
	return c/c.w;
}

// int2 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
CUDA_VEC_HOSTDEVICE int2 make_int2(int const * a)
{
    return make_int2(a[0], a[1]);
}
CUDA_VEC_HOSTDEVICE int2 make_int2(int s)
{
    return make_int2(s, s);
}
CUDA_VEC_HOSTDEVICE int2 make_int2(int3 const & a)
{
    return make_int2(a.x, a.y); // discards z
}
CUDA_VEC_HOSTDEVICE int2 make_int2(int4 const & a)
{
    return make_int2(a.x, a.y); // discards z, w
}
CUDA_VEC_HOSTDEVICE int2 make_int2(float2 const & a)
{
    return make_int2(int(a.x), int(a.y));
}
CUDA_VEC_HOSTDEVICE int2 make_int2(uint2 const & a)
{
    return make_int2(int(a.x), int(a.y));
}

// negate
CUDA_VEC_HOSTDEVICE int2 operator-(int2 const & a)
{
    return make_int2(-a.x, -a.y);
}

// min
CUDA_VEC_HOSTDEVICE int2 min(int2 const & a, int2 const & b)
{
    return make_int2(min(a.x, b.x), min(a.y, b.y));
}

// max
CUDA_VEC_HOSTDEVICE int2 max(int2 const & a, int2 const & b)
{
    return make_int2(max(a.x, b.x), max(a.y, b.y));
}

// addition
CUDA_VEC_HOSTDEVICE int2 operator+(int2 const & a, int2 const & b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
CUDA_VEC_HOSTDEVICE int2 operator+(int2 const & a, int b)
{
    return make_int2(a.x + b, a.y + b);
}
CUDA_VEC_HOSTDEVICE int2 operator+(int a, int2 const & b)
{
    return make_int2(a + b.x, a + b.y);
}
CUDA_VEC_HOSTDEVICE void operator+=(int2 & a, int2 const & b)
{
    a.x += b.x; a.y += b.y;
}
CUDA_VEC_HOSTDEVICE void operator+=(int2 & a, int b)
{
    a.x += b; a.y += b;
}

// subtract
CUDA_VEC_HOSTDEVICE int2 operator-(int2 const & a, int2 const & b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
CUDA_VEC_HOSTDEVICE int2 operator-(int2 const & a, int b)
{
    return make_int2(a.x - b, a.y - b);
}
CUDA_VEC_HOSTDEVICE int2 operator-(int a, int2 const & b)
{
    return make_int2(a - b.x, a - b.y);
}
CUDA_VEC_HOSTDEVICE void operator-=(int2 & a, int2 const & b)
{
    a.x -= b.x; a.y -= b.y;
}
CUDA_VEC_HOSTDEVICE void operator-=(int2 & a, int b)
{
    a.x -= b; a.y -= b;
}

// multiply
CUDA_VEC_HOSTDEVICE int2 operator*(int2 const & a, int2 const & b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
CUDA_VEC_HOSTDEVICE int2 operator*(int2 const & a, int b)
{
    return make_int2(a.x * b, a.y * b);
}
CUDA_VEC_HOSTDEVICE int2 operator*(int a, int2 const & b)
{
    return make_int2(a * b.x, a * b.y);
}
CUDA_VEC_HOSTDEVICE void operator*=(int2 & a, int2 const & b)
{
    a.x *= b.x; a.y *= b.y;
}
CUDA_VEC_HOSTDEVICE void operator*=(int2 & a, int b)
{
    a.x *= b; a.y *= b;
}

// divide
CUDA_VEC_HOSTDEVICE int2 operator/(int2 const & a, int2 const & b)
{
    return make_int2(a.x / b.x, a.y / b.y);
}
CUDA_VEC_HOSTDEVICE int2 operator/(int2 const & a, int b)
{
    return make_int2(a.x / b, a.y / b);
}
CUDA_VEC_HOSTDEVICE int2 operator/(int a, int2 const & b)
{
    return make_int2(a / b.x, a / b.y);
}
CUDA_VEC_HOSTDEVICE void operator/=(int2 & a, int2 const & b)
{
    a.x /= b.x; a.y /= b.y;
}
CUDA_VEC_HOSTDEVICE void operator/=(int2 & a, int b)
{
    a.x /= b; a.y /= b;
}

// modulo
CUDA_VEC_HOSTDEVICE int2 operator%(int2 const & a, int2 const & b)
{
    return make_int2(a.x % b.x, a.y % b.y);
}
CUDA_VEC_HOSTDEVICE int2 operator%(int2 const & a, int b)
{
    return make_int2(a.x % b, a.y % b);
}
CUDA_VEC_HOSTDEVICE int2 operator%(int a, int2 const & b)
{
    return make_int2(a % b.x, a % b.y);
}
CUDA_VEC_HOSTDEVICE void operator%=(int2 & a, int2 const & b)
{
    a.x %= b.x; a.y %= b.y;
}
CUDA_VEC_HOSTDEVICE void operator%=(int2 & a, int b)
{
    a.x %= b; a.y %= b;
}

// clamp
CUDA_VEC_HOSTDEVICE int2 clamp(int2 const & v, int a, int b)
{
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
CUDA_VEC_HOSTDEVICE int2 clamp(int2 const & v, int2 const & a, int2 const & b)
{
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

// int3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
CUDA_VEC_HOSTDEVICE int3 make_int3(int const * a)
{
    return make_int3(a[0], a[1], a[2]);
}
CUDA_VEC_HOSTDEVICE int3 make_int3(int s)
{
    return make_int3(s, s, s);
}
CUDA_VEC_HOSTDEVICE int3 make_int3(int2 const & a)
{
    return make_int3(a.x, a.y, 0);
}
CUDA_VEC_HOSTDEVICE int3 make_int3(int2 const & a, int s)
{
    return make_int3(a.x, a.y, s);
}
CUDA_VEC_HOSTDEVICE int3 make_int3(int4 const & a)
{
    return make_int3(a.x, a.y, a.z);  // discards w
}
CUDA_VEC_HOSTDEVICE int3 make_int3(float3 const & a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}
CUDA_VEC_HOSTDEVICE int3 make_int3(uint3 const & a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

// negate
CUDA_VEC_HOSTDEVICE int3 operator-(int3 const & a)
{
    return make_int3(-a.x, -a.y, -a.z);
}

// min
CUDA_VEC_HOSTDEVICE int3 min(int3 const & a, int3 const & b)
{
    return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

// max
CUDA_VEC_HOSTDEVICE int3 max(int3 const & a, int3 const & b)
{
    return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

// addition
CUDA_VEC_HOSTDEVICE int3 operator+(int3 const & a, int3 const & b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
CUDA_VEC_HOSTDEVICE int3 operator+(int3 const & a, int b)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
CUDA_VEC_HOSTDEVICE int3 operator+(int a, int3 const & b)
{
    return make_int3(a + b.x, a + b.y, a + b.z);
}
CUDA_VEC_HOSTDEVICE void operator+=(int3 & a, int3 const & b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
CUDA_VEC_HOSTDEVICE void operator+=(int3 & a, int b)
{
    a.x += b; a.y += b; a.z += b;
}

// subtract
CUDA_VEC_HOSTDEVICE int3 operator-(int3 const & a, int3 const & b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
CUDA_VEC_HOSTDEVICE int3 operator-(int3 const & a, int b)
{
    return make_int3(a.x - b, a.y - b, a.z - b);
}
CUDA_VEC_HOSTDEVICE int3 operator-(int a, int3 const & b)
{
    return make_int3(a - b.x, a - b.y, a - b.z);
}
CUDA_VEC_HOSTDEVICE void operator-=(int3 & a, int3 const & b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
CUDA_VEC_HOSTDEVICE void operator-=(int3 & a, int b)
{
    a.x -= b; a.y -= b; a.z -= b;
}

// multiply
CUDA_VEC_HOSTDEVICE int3 operator*(int3 const & a, int3 const & b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
CUDA_VEC_HOSTDEVICE int3 operator*(int3 const & a, int b)
{
    return make_int3(a.x * b, a.y * b, a.z * b);
}
CUDA_VEC_HOSTDEVICE int3 operator*(int a, int3 const & b)
{
    return make_int3(a * b.x, a * b.y, a * b.z);
}
CUDA_VEC_HOSTDEVICE void operator*=(int3 & a, int3 const & b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}
CUDA_VEC_HOSTDEVICE void operator*=(int3 & a, int b)
{
    a.x *= b; a.y *= b; a.z *= b;
}

// divide
CUDA_VEC_HOSTDEVICE int3 operator/(int3 const & a, int3 const & b)
{
    return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
CUDA_VEC_HOSTDEVICE int3 operator/(int3 const & a, int b)
{
    return make_int3(a.x / b, a.y / b, a.z / b);
}
CUDA_VEC_HOSTDEVICE int3 operator/(int a, int3 const & b)
{
    return make_int3(a / b.x, a / b.y, a / b.z);
}
CUDA_VEC_HOSTDEVICE void operator/=(int3 & a, int3 const & b)
{
    a.x /= b.x; a.y /= b.y; a.z /= b.z;
}
CUDA_VEC_HOSTDEVICE void operator/=(int3 & a, int b)
{
    a.x /= b; a.y /= b; a.z /= b;
}

// modulo
CUDA_VEC_HOSTDEVICE int3 operator%(int3 const & a, int3 const & b)
{
    return make_int3(a.x % b.x, a.y % b.y, a.z % b.z);
}
CUDA_VEC_HOSTDEVICE int3 operator%(int3 const & a, int b)
{
    return make_int3(a.x % b, a.y % b, a.z % b);
}
CUDA_VEC_HOSTDEVICE int3 operator%(int a, int3 const & b)
{
    return make_int3(a % b.x, a % b.y, a % b.z);
}
CUDA_VEC_HOSTDEVICE void operator%=(int3 & a, int3 const & b)
{
    a.x %= b.x; a.y %= b.y; a.z %= b.z;
}
CUDA_VEC_HOSTDEVICE void operator%=(int3 & a, int b)
{
    a.x %= b; a.y %= b; a.z %= b;
}

// clamp
CUDA_VEC_HOSTDEVICE int3 clamp(int3 const & v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
CUDA_VEC_HOSTDEVICE int3 clamp(int3 const & v, int3 const & a, int3 const & b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}


// int4 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
CUDA_VEC_HOSTDEVICE int4 make_int4(int const * a)
{
    return make_int4(a[0], a[1], a[2], a[3]);
}
CUDA_VEC_HOSTDEVICE int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}
CUDA_VEC_HOSTDEVICE int4 make_int4(int2 const & a)
{
    return make_int4(a.x, a.y, 0, 0);
}
CUDA_VEC_HOSTDEVICE int4 make_int4(int3 const & a)
{
    return make_int4(a.x, a.y, a.z, 0);
}
CUDA_VEC_HOSTDEVICE int4 make_int4(int2 const & a, int z, int w)
{
    return make_int4(a.x, a.y, z, w);
}
CUDA_VEC_HOSTDEVICE int4 make_int4(int3 const & a, int w)
{
    return make_int4(a.x, a.y, a.z, w);
}
CUDA_VEC_HOSTDEVICE int4 make_int4(float4 const & a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
CUDA_VEC_HOSTDEVICE int4 make_int4(uint4 const & a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}

// negate
CUDA_VEC_HOSTDEVICE int4 operator-(int4 const & a)
{
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

// min
CUDA_VEC_HOSTDEVICE int4 min(int4 const & a, int4 const & b)
{
    return make_int4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

// max
CUDA_VEC_HOSTDEVICE int4 max(int4 const & a, int4 const & b)
{
    return make_int4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

// addition
CUDA_VEC_HOSTDEVICE int4 operator+(int4 const & a, int4 const & b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
CUDA_VEC_HOSTDEVICE int4 operator+(int4 const & a, int b)
{
    return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
CUDA_VEC_HOSTDEVICE int4 operator+(int a, int4 const & b)
{
    return make_int4(a + b.x, a + b.y, a + b.z, a + b.w);
}
CUDA_VEC_HOSTDEVICE void operator+=(int4 & a, int4 const & b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
CUDA_VEC_HOSTDEVICE void operator+=(int4 & a, int b)
{
    a.x += b; a.y += b; a.z += b; a.w += b;
}

// subtract
CUDA_VEC_HOSTDEVICE int4 operator-(int4 const & a, int4 const & b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
CUDA_VEC_HOSTDEVICE int4 operator-(int4 const & a, int b)
{
    return make_int4(a.x - b, a.y - b, a.z - b, a.w - b);
}
CUDA_VEC_HOSTDEVICE int4 operator-(int a, int4 const & b)
{
    return make_int4(a - b.x, a - b.y, a - b.z, a - b.w);
}
CUDA_VEC_HOSTDEVICE void operator-=(int4 & a, int4 const & b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
CUDA_VEC_HOSTDEVICE void operator-=(int4 & a, int b)
{
    a.x -= b; a.y -= b; a.z -= b; a.w -= b;
}

// multiply
CUDA_VEC_HOSTDEVICE int4 operator*(int4 const & a, int4 const & b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
CUDA_VEC_HOSTDEVICE int4 operator*(int4 const & a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b, a.w * b);
}
CUDA_VEC_HOSTDEVICE int4 operator*(int a, int4 const & b)
{
    return make_int4(a * b.x, a * b.y, a * b.z, a * b.w);
}
CUDA_VEC_HOSTDEVICE void operator*=(int4 & a, int4 const &  b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}
CUDA_VEC_HOSTDEVICE void operator*=(int4 & a, int b)
{
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

// divide
CUDA_VEC_HOSTDEVICE int4 operator/(int4 const & a, int4 const & b)
{
    return make_int4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
CUDA_VEC_HOSTDEVICE int4 operator/(int4 const & a, int b)
{
    return make_int4(a.x / b, a.y / b, a.z / b, a.w / b);
}
CUDA_VEC_HOSTDEVICE int4 operator/(int a, int4 const & b)
{
    return make_int4(a / b.x, a / b.y, a / b.z, a / b.w);
}
CUDA_VEC_HOSTDEVICE void operator/=(int4 & a, int4 const & b)
{
    a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
}
CUDA_VEC_HOSTDEVICE void operator/=(int4 & a, int b)
{
    a.x /= b; a.y /= b; a.z /= b; a.w /= b;
}

// modulo
CUDA_VEC_HOSTDEVICE int4 operator%(int4 const & a, int4 const & b)
{
    return make_int4(a.x % b.x, a.y % b.y, a.z % b.z, a.w % b.w);
}
CUDA_VEC_HOSTDEVICE int4 operator%(int4 const & a, int b)
{
    return make_int4(a.x % b, a.y % b, a.z % b, a.w % b);
}
CUDA_VEC_HOSTDEVICE int4 operator%(int a, int4 const & b)
{
    return make_int4(a % b.x, a % b.y, a % b.z, a % b.w);
}
CUDA_VEC_HOSTDEVICE void operator%=(int4 & a, int4 const & b)
{
    a.x %= b.x; a.y %= b.y; a.z %= b.z; a.w %= b.w;
}
CUDA_VEC_HOSTDEVICE void operator%=(int4 & a, int b)
{
    a.x %= b; a.y %= b; a.z %= b; a.w %= b;
}

// clamp
CUDA_VEC_HOSTDEVICE int4 clamp(int4 const & v, int a, int b)
{
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
CUDA_VEC_HOSTDEVICE int4 clamp(int4 const & v, int4 const & a, int4 const & b)
{
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

// uint2 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
CUDA_VEC_HOSTDEVICE uint2 make_uint2(unsigned int const * a)
{
    return make_uint2(a[0], a[1]);
}
CUDA_VEC_HOSTDEVICE uint2 make_uint2(unsigned int s)
{
    return make_uint2(s, s);
}
CUDA_VEC_HOSTDEVICE uint2 make_uint2(uint3 const & a)
{
    return make_uint2(a.x, a.y); // discards z
}
CUDA_VEC_HOSTDEVICE uint2 make_uint2(uint4 const & a)
{
    return make_uint2(a.x, a.y); // discards z, w
}
CUDA_VEC_HOSTDEVICE uint2 make_uint2(float2 const & a)
{
    return make_uint2((unsigned int)a.x, (unsigned int)a.y);
}
CUDA_VEC_HOSTDEVICE uint2 make_uint2(int2 const & a)
{
	return make_uint2((unsigned int)a.x, (unsigned int)a.y);
}

// min
CUDA_VEC_HOSTDEVICE uint2 min(uint2 const & a, uint2 const & b)
{
    return make_uint2(min(a.x, b.x), min(a.y, b.y));
}

// max
CUDA_VEC_HOSTDEVICE uint2 max(uint2 const & a, uint2 const & b)
{
    return make_uint2(max(a.x, b.x), max(a.y, b.y));
}

// addition
CUDA_VEC_HOSTDEVICE uint2 operator+(uint2 const & a, uint2 const & b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}
CUDA_VEC_HOSTDEVICE uint2 operator+(uint2 const & a, unsigned int b)
{
    return make_uint2(a.x + b, a.y + b);
}
CUDA_VEC_HOSTDEVICE uint2 operator+(unsigned int a, uint2 const & b)
{
    return make_uint2(a + b.x, a + b.y);
}
CUDA_VEC_HOSTDEVICE void operator+=(uint2 & a, uint2 b)
{
    a.x += b.x; a.y += b.y;
}
CUDA_VEC_HOSTDEVICE void operator+=(uint2 & a, unsigned int b)
{
    a.x += b; a.y += b;
}

// subtract
CUDA_VEC_HOSTDEVICE uint2 operator-(uint2 const & a, uint2 const & b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}
CUDA_VEC_HOSTDEVICE uint2 operator-(uint2 const & a, unsigned int b)
{
    return make_uint2(a.x - b, a.y - b);
}
CUDA_VEC_HOSTDEVICE uint2 operator-(unsigned int a, uint2 const & b)
{
    return make_uint2(a - b.x, a - b.y);
}
CUDA_VEC_HOSTDEVICE void operator-=(uint2 & a, uint2 b)
{
    a.x -= b.x; a.y -= b.y;
}
CUDA_VEC_HOSTDEVICE void operator-=(uint2 & a, unsigned int b)
{
    a.x -= b; a.y -= b;
}

// multiply
CUDA_VEC_HOSTDEVICE uint2 operator*(uint2 const & a, uint2 const & b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}
CUDA_VEC_HOSTDEVICE uint2 operator*(uint2 const & a, unsigned int b)
{
    return make_uint2(a.x * b, a.y * b);
}
CUDA_VEC_HOSTDEVICE uint2 operator*(unsigned int a, uint2 const & b)
{
    return make_uint2(a * b.x, a * b.y);
}
CUDA_VEC_HOSTDEVICE void operator*=(uint2 & a, uint2 const & b)
{
    a.x *= b.x; a.y *= b.y;
}
CUDA_VEC_HOSTDEVICE void operator*=(uint2 & a, unsigned int b)
{
    a.x *= b; a.y *= b;
}

// divide
CUDA_VEC_HOSTDEVICE uint2 operator/(uint2 const & a, uint2 const & b)
{
    return make_uint2(a.x / b.x, a.y / b.y);
}
CUDA_VEC_HOSTDEVICE uint2 operator/(uint2 const & a, unsigned int b)
{
    return make_uint2(a.x / b, a.y / b);
}
CUDA_VEC_HOSTDEVICE uint2 operator/(unsigned int a, uint2 const & b)
{
    return make_uint2(a / b.x, a / b.y);
}
CUDA_VEC_HOSTDEVICE void operator/=(uint2 & a, uint2 const & b)
{
    a.x /= b.x; a.y /= b.y;
}
CUDA_VEC_HOSTDEVICE void operator/=(uint2 & a, unsigned int b)
{
    a.x /= b; a.y /= b;
}

// modulo
CUDA_VEC_HOSTDEVICE uint2 operator%(uint2 const & a, uint2 const & b)
{
    return make_uint2(a.x % b.x, a.y % b.y);
}
CUDA_VEC_HOSTDEVICE uint2 operator%(uint2 const & a, unsigned int b)
{
    return make_uint2(a.x % b, a.y % b);
}
CUDA_VEC_HOSTDEVICE uint2 operator%(unsigned int a, uint2 const & b)
{
    return make_uint2(a % b.x, a % b.y);
}
CUDA_VEC_HOSTDEVICE void operator%=(uint2 & a, uint2 const & b)
{
    a.x %= b.x; a.y %= b.y;
}
CUDA_VEC_HOSTDEVICE void operator%=(uint2 & a, unsigned int b)
{
    a.x %= b; a.y %= b;
}

// clamp
CUDA_VEC_HOSTDEVICE uint2 clamp(uint2 const & v, unsigned int a, unsigned int b)
{
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
CUDA_VEC_HOSTDEVICE uint2 clamp(uint2 const & v, uint2 const & a, uint2 const & b)
{
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

// uint3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
CUDA_VEC_HOSTDEVICE uint3 make_uint3(unsigned int const * a)
{
    return make_uint3(a[0], a[1], a[2]);
}
CUDA_VEC_HOSTDEVICE uint3 make_uint3(unsigned int s)
{
    return make_uint3(s, s, s);
}
CUDA_VEC_HOSTDEVICE uint3 make_uint3(uint2 const & a)
{
    return make_uint3(a.x, a.y, 0);
}
CUDA_VEC_HOSTDEVICE uint3 make_uint3(uint2 const & a, unsigned int s)
{
    return make_uint3(a.x, a.y, s);
}
CUDA_VEC_HOSTDEVICE uint3 make_uint3(uint4 const & a)
{
    return make_uint3(a.x, a.y, a.z);  // discards w
}
CUDA_VEC_HOSTDEVICE uint3 make_uint3(float3 const & a)
{
    return make_uint3((unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z);
}
CUDA_VEC_HOSTDEVICE uint3 make_uint3(int3 const & a)
{
	return make_uint3((unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z);
}


// min
CUDA_VEC_HOSTDEVICE uint3 min(uint3 const & a, uint3 const & b)
{
    return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

// max
CUDA_VEC_HOSTDEVICE uint3 max(uint3 const & a, uint3 const & b)
{
    return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

// addition
CUDA_VEC_HOSTDEVICE uint3 operator+(uint3 const & a, uint3 const & b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
CUDA_VEC_HOSTDEVICE uint3 operator+(uint3 const & a, unsigned int b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
CUDA_VEC_HOSTDEVICE uint3 operator+(unsigned int a, uint3 const & b)
{
    return make_uint3(a + b.x, a + b.y, a + b.z);
}
CUDA_VEC_HOSTDEVICE void operator+=(uint3 & a, uint3 const & b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
CUDA_VEC_HOSTDEVICE void operator+=(uint3 & a, unsigned int b)
{
    a.x += b; a.y += b; a.z += b;
}

// subtract
CUDA_VEC_HOSTDEVICE uint3 operator-(uint3 const & a, uint3 const & b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
CUDA_VEC_HOSTDEVICE uint3 operator-(uint3 const & a, unsigned int b)
{
    return make_uint3(a.x - b, a.y - b, a.z - b);
}
CUDA_VEC_HOSTDEVICE uint3 operator-(unsigned int a, uint3 const & b)
{
    return make_uint3(a - b.x, a - b.y, a - b.z);
}
CUDA_VEC_HOSTDEVICE void operator-=(uint3 & a, uint3 const & b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
CUDA_VEC_HOSTDEVICE void operator-=(uint3 & a, unsigned int b)
{
    a.x -= b; a.y -= b; a.z -= b;
}

// multiply
CUDA_VEC_HOSTDEVICE uint3 operator*(uint3 const & a, uint3 const & b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
CUDA_VEC_HOSTDEVICE uint3 operator*(uint3 const & a, unsigned int b)
{
    return make_uint3(a.x * b, a.y * b, a.z * b);
}
CUDA_VEC_HOSTDEVICE uint3 operator*(unsigned int a, uint3 const & b)
{
    return make_uint3(a * b.x, a * b.y, a * b.z);
}
CUDA_VEC_HOSTDEVICE void operator*=(uint3 & a, uint3 const & b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}
CUDA_VEC_HOSTDEVICE void operator*=(uint3 & a, unsigned int b)
{
    a.x *= b; a.y *= b; a.z *= b;
}

// divide
CUDA_VEC_HOSTDEVICE uint3 operator/(uint3 const & a, uint3 const & b)
{
    return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}
CUDA_VEC_HOSTDEVICE uint3 operator/(uint3 const & a, unsigned int b)
{
    return make_uint3(a.x / b, a.y / b, a.z / b);
}
CUDA_VEC_HOSTDEVICE uint3 operator/(unsigned int a, uint3 const & b)
{
    return make_uint3(a / b.x, a / b.y, a / b.z);
}
CUDA_VEC_HOSTDEVICE void operator/=(uint3 & a, uint3 const & b)
{
    a.x /= b.x; a.y /= b.y; a.z /= b.z;
}
CUDA_VEC_HOSTDEVICE void operator/=(uint3 & a, unsigned int b)
{
    a.x /= b; a.y /= b; a.z /= b;
}

// modulo
CUDA_VEC_HOSTDEVICE uint3 operator%(uint3 const & a, uint3 const & b)
{
    return make_uint3(a.x % b.x, a.y % b.y, a.z % b.z);
}
CUDA_VEC_HOSTDEVICE uint3 operator%(uint3 const & a, unsigned int b)
{
    return make_uint3(a.x % b, a.y % b, a.z % b);
}
CUDA_VEC_HOSTDEVICE uint3 operator%(unsigned int a, uint3 const & b)
{
    return make_uint3(a % b.x, a % b.y, a % b.z);
}
CUDA_VEC_HOSTDEVICE void operator%=(uint3 & a, uint3 const & b)
{
    a.x %= b.x; a.y %= b.y; a.z %= b.z;
}
CUDA_VEC_HOSTDEVICE void operator%=(uint3 & a, unsigned int b)
{
    a.x %= b; a.y %= b; a.z %= b;
}

// clamp
CUDA_VEC_HOSTDEVICE uint3 clamp(uint3 const & v, unsigned int a, unsigned int b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
CUDA_VEC_HOSTDEVICE uint3 clamp(uint3 const & v, uint3 const & a, uint3 const & b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// uint4 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
CUDA_VEC_HOSTDEVICE uint4 make_uint4(unsigned int const * a)
{
    return make_uint4(a[0], a[1], a[2], a[3]);
}
CUDA_VEC_HOSTDEVICE uint4 make_uint4(unsigned int s)
{
    return make_uint4(s, s, s, s);
}
CUDA_VEC_HOSTDEVICE uint4 make_uint4(uint2 const & a)
{
    return make_uint4(a.x, a.y, 0, 0);
}
CUDA_VEC_HOSTDEVICE uint4 make_uint4(uint3 const & a)
{
    return make_uint4(a.x, a.y, a.z, 0);
}
CUDA_VEC_HOSTDEVICE uint4 make_uint4(uint2 const & a, unsigned int z, unsigned int w)
{
    return make_uint4(a.x, a.y, z, w);
}
CUDA_VEC_HOSTDEVICE uint4 make_uint4(uint3 const & a, unsigned int w)
{
    return make_uint4(a.x, a.y, a.z, w);
}
CUDA_VEC_HOSTDEVICE uint4 make_uint4(float4 const & a)
{
    return make_uint4((unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z, (unsigned int)a.w);
}
CUDA_VEC_HOSTDEVICE uint4 make_uint4(int4 const & a)
{
	return make_uint4((unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z, (unsigned int)a.w);
}

// min
CUDA_VEC_HOSTDEVICE uint4 min(uint4 const & a, uint4 const & b)
{
    return make_uint4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

// max
CUDA_VEC_HOSTDEVICE uint4 max(uint4 const & a, uint4 const & b)
{
    return make_uint4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

// addition
CUDA_VEC_HOSTDEVICE uint4 operator+(uint4 const & a, uint4 const & b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
CUDA_VEC_HOSTDEVICE uint4 operator+(uint4 const & a, int b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
CUDA_VEC_HOSTDEVICE uint4 operator+(int a, uint4 const & b)
{
    return make_uint4(a + b.x, a + b.y, a + b.z, a + b.w);
}
CUDA_VEC_HOSTDEVICE void operator+=(uint4 & a, uint4 const & b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
CUDA_VEC_HOSTDEVICE void operator+=(uint4 & a, unsigned int b)
{
    a.x += b; a.y += b; a.z += b; a.w += b;
}

// subtract
CUDA_VEC_HOSTDEVICE uint4 operator-(uint4 const & a, uint4 const & b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
CUDA_VEC_HOSTDEVICE uint4 operator-(uint4 const & a, int b)
{
    return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b);
}
CUDA_VEC_HOSTDEVICE uint4 operator-(int a, uint4 const & b)
{
    return make_uint4(a - b.x, a - b.y, a - b.z, a - b.w);
}
CUDA_VEC_HOSTDEVICE void operator-=(uint4 & a, uint4 const & b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
CUDA_VEC_HOSTDEVICE void operator-=(uint4 & a, unsigned int b)
{
    a.x -= b; a.y -= b; a.z -= b; a.w -= b;
}

// multiply
CUDA_VEC_HOSTDEVICE uint4 operator*(uint4 const & a, uint4 const & b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
CUDA_VEC_HOSTDEVICE uint4 operator*(uint4 const & a, int b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b);
}
CUDA_VEC_HOSTDEVICE uint4 operator*(int a, uint4 const & b)
{
    return make_uint4(a * b.x, a * b.y, a * b.z, a * b.w);
}
CUDA_VEC_HOSTDEVICE void operator*=(uint4 & a, uint4 const & b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}
CUDA_VEC_HOSTDEVICE void operator*=(uint4 & a, unsigned int b)
{
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

// divide
CUDA_VEC_HOSTDEVICE uint4 operator/(uint4 const & a, uint4 const & b)
{
    return make_uint4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
CUDA_VEC_HOSTDEVICE uint4 operator/(uint4 const & a, int b)
{
    return make_uint4(a.x / b, a.y / b, a.z / b, a.w / b);
}
CUDA_VEC_HOSTDEVICE uint4 operator/(int a, uint4 const & b)
{
    return make_uint4(a / b.x, a / b.y, a / b.z, a / b.w);
}
CUDA_VEC_HOSTDEVICE void operator/=(uint4 & a, uint4 const & b)
{
    a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
}
CUDA_VEC_HOSTDEVICE void operator/=(uint4 & a, unsigned int b)
{
    a.x /= b; a.y /= b; a.z /= b; a.w /= b;
}

// modulo
CUDA_VEC_HOSTDEVICE uint4 operator%(uint4 const &  a, uint4 const &  b)
{
    return make_uint4(a.x % b.x, a.y % b.y, a.z % b.z, a.w % b.w);
}
CUDA_VEC_HOSTDEVICE uint4 operator%(uint4 const &  a, int b)
{
    return make_uint4(a.x % b, a.y % b, a.z % b, a.w % b);
}
CUDA_VEC_HOSTDEVICE uint4 operator%(int a, uint4 const &  b)
{
    return make_uint4(a % b.x, a % b.y, a % b.z, a % b.w);
}
CUDA_VEC_HOSTDEVICE void operator%=(uint4 & a, uint4 const & b)
{
    a.x %= b.x; a.y %= b.y; a.z %= b.z; a.w %= b.w;
}
CUDA_VEC_HOSTDEVICE void operator%=(uint4 & a, unsigned int b)
{
    a.x %= b; a.y %= b; a.z %= b; a.w %= b;
}

// clamp
CUDA_VEC_HOSTDEVICE uint4 clamp(uint4 const & v, unsigned int a, unsigned int b)
{
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
CUDA_VEC_HOSTDEVICE uint4 clamp(uint4 const & v, uint4 const & a, uint4 const & b)
{
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

//#ifndef __CUDACC__
//
//#include <cmath>
//#include <iostream>
//#include <cassert>
//#include <algorithm>
//
//#else
//
//#endif
//
//#endif

// float3x3 class
////////////////////////////////////////////////////////////////////////////////

class float3x3
{
	float val[3*3];

public:

#ifndef __CUDACC__ // CUDA doesn't support non-empty constructors

	/// @name Constructors
	// @{
	/// Default constructor
	float3x3()
	{}

	/// Copy constructor
	float3x3(float3x3 const & m)
	{
		for(int k=0; k<3*3; ++k) val[k] = m[k];
	}

	/// Copy constructor with a given array (must have a length >= 3*3)
	float3x3(float const * m)
	{
		for(int k=0; k<3*3; ++k) val[k] = m[k];
	}

	/// Constructor with an initial value for all elements
	explicit float3x3(float const & w)
	{
		for(int k=0; k<3*3; ++k) val[k] = w;
	}
	// @}

#endif

	/// @name Array subscripting, matrix access and dereferencing operators
	// @{
	/// Array subscripting operator
	CUDA_VEC_HOSTDEVICE float & operator[](int k)
	{
		//CUDA_VEC_ASSERT( (k>=0) && (k<3*3) );
		return val[k];
	}

	/// Constant array subscripting operator.
	CUDA_VEC_HOSTDEVICE const float & operator[](int k) const
	{
		//CUDA_VEC_ASSERT( (k>=0) && (k<3*3) );
		return val[k];
	}

	/// Matrix access
	CUDA_VEC_HOSTDEVICE float & get(int k, int l)
	{
		CUDA_VEC_ASSERT( (k>=0) && (l>=0) && (k<3) && (l<3) );
		return val[k*3+l];
	}

	/// Matrix access
	CUDA_VEC_HOSTDEVICE const float & get(int k, int l) const
	{
		CUDA_VEC_ASSERT( (k>=0) && (l>=0) && (k<3) && (l<3) );
		return val[k*3+l];
	}

	/// Matrix access to column vectors
	CUDA_VEC_HOSTDEVICE float3 getCol(int l) const
	{
		CUDA_VEC_ASSERT( (l>=0) && (l<3) );
		float3 v;
		for(int k=0; k<3; ++k)
			(&v.x)[k] = val[k*3+l];
		return v;
	}

	/// Matrix access to row vectors
	CUDA_VEC_HOSTDEVICE float3 getRow(int k) const
	{
		CUDA_VEC_ASSERT( (k>=0) && (k<3) );
        int r = k*3;
		return make_float3(val[r], val[r+1], val[r+2]);
	}


	/// Dereferencing operator
	CUDA_VEC_HOSTDEVICE operator float * ()
	{
		return val;
	}

	/// Constant dereferencing operator
	CUDA_VEC_HOSTDEVICE operator const float * () const
	{
		return val;
	}
	// @}

	/// @name Assignment operator and arithmetic assignment operators
	// @{
	/// Assignmet operator
	CUDA_VEC_HOSTDEVICE float3x3 & operator=(float3x3 const & m)
	{
		for(int k=0; k<3*3; ++k) val[k] = m[k];
		return (*this);
	}

	/// Add and assign
	CUDA_VEC_HOSTDEVICE float3x3 & operator+=(float3x3 const & m)
	{
		for(int k=0; k<3*3; ++k) val[k] += m[k];
		return (*this);
	}

	/// Subtract and assign
	CUDA_VEC_HOSTDEVICE float3x3 & operator-=(float3x3 const & m)
	{
		for(int k=0; k<3*3; ++k) val[k] -= m[k];
		return (*this);
	}

	/// Multiply a scalar and assign
	CUDA_VEC_HOSTDEVICE float3x3 & operator*=(float w)
	{
		for(int k=0; k<3*3; ++k) val[k] *= w;
		return (*this);
	}

	/// Multiply a matrix and assign
	CUDA_VEC_HOSTDEVICE float3x3 & operator*=(float3x3 const & m)
	{
		float3x3 result;
		for(int i=0; i<3; ++i)
			for(int j=0; j<3; ++j){
				float sum(0);
				for(int k=0; k<3; k++)
					sum += val[i*3 + k] * m[k*3 + j];
				result[i*3 + j] = sum;
				}
		*this = result;
		return *this;
	}

	/// Divide by a scalar and assign
	CUDA_VEC_HOSTDEVICE float3x3 & operator/=(float w)
	{
		for(int k=0; k<3*3; ++k) val[k] /= w;
		return (*this);
	}

	///// Modulo by a scalar and assign
	//float3x3 &operator%=(const float &w)
	//{
	//	for(int k=0; k<3*3; ++k) val[k] %= w;
	//	return (*this);
	//}

	/// Sum of two matrices
	CUDA_VEC_HOSTDEVICE float3x3 operator+(float3x3 const & m) const
	{
		float3x3 res;
		for(int k=0; k<3*3; ++k) res[k] = val[k] + m[k];
		return res;
	}

	/// Difference of two matrices
	CUDA_VEC_HOSTDEVICE float3x3 operator-(float3x3 const & m) const
	{
		float3x3 res;
		for(int k=0; k<3*3; ++k) res[k] = val[k] - m[k];
		return res;
	}

	/// Multiply matrix by scalar
	CUDA_VEC_HOSTDEVICE float3x3 operator*(float const & w) const
	{
		float3x3 res;
		for(int k=0; k<3*3; ++k) res[k] = val[k] * w;
		return res;
	}

	/// Product of matrix and vector
	CUDA_VEC_HOSTDEVICE const float3 operator*(float3 const & v) const
	{
        return make_float3(
            val[0] * v.x + val[1] * v.y + val[2] * v.z,
            val[3] * v.x + val[4] * v.y + val[5] * v.z,
            val[6] * v.x + val[7] * v.y + val[8] * v.z);
	}

	/// Product of two matrices
	CUDA_VEC_HOSTDEVICE float3x3 operator*(float3x3 const & m) const
	{
		float3x3 res;
		for(int i=0; i<3; ++i)
			for(int j=0; j<3; ++j){
				float sum(0);
				for(int k=0; k<3; k++)
					sum += val[i*3 + k] * m[k*3 + j];
				res[i*3 + j] = sum;
				}
		return res;
	}

	/// Divide matrix by scalar
	CUDA_VEC_HOSTDEVICE float3x3 operator/(float w) const
	{
		float3x3 res;
		for(int k=0; k<3*3; ++k) res[k] = val[k] / w;
		return res;
	}

	/// Unary -
	CUDA_VEC_HOSTDEVICE float3x3 operator-() const
	{
		float3x3 res;
		for(int k=0; k<3*3; ++k) res[k] = -val[k];
		return res;
	}

	// @}

	/// @name Matrix functions
	// @{

	/// Clear the matrix to zero
	CUDA_VEC_HOSTDEVICE void clear()
	{
		for(int k=0; k<3*3; ++k)
			val[k] = float(0);
	}

	/// Multiply with matrix B and store result in result
	CUDA_VEC_HOSTDEVICE float3x3 & multMat(float3x3 const & B, float3x3 & result) const
	{
		for(int i=0; i<3; ++i)
			for(int j=0; j<3; ++j){
				float sum(0);
				for(int k=0; k<3; k++)
					sum += val[i*3 + k] * B[k*3 + j];
				result[i*3 + j] = sum;
				}
		return result;
	}


	/// Multiply with matrix B and accumulate result to matrix result
	CUDA_VEC_HOSTDEVICE float3x3 & multMatAdd(float3x3 const & B, float3x3 & result) const
	{
		for(int i=0; i<3; ++i)
			for(int j=0; j<3; ++j){
				float sum(0);
				for(int k=0; k<3; k++)
					sum += val[i*3 + k] * B[k*3 + j];
				result[i*3 + j] += sum;
				}
		return result;
	}


	/// Multiply with transposed matrix B and store result in result
	CUDA_VEC_HOSTDEVICE float3x3 & multMatT(float3x3 const & B, float3x3 & result) const
	{
		for(int i=0; i<3; ++i)
			for(int j=0; j<3; ++j){
				float sum(0);
				for(int k=0; k<3; k++)
					sum += val[i*3 + k] * B[j*3 + k];
				result[i*3 + j] = sum;
				}
		return result;
	}

	/// Multiply with transposed matrix B and accumulate result to matrix result
	CUDA_VEC_HOSTDEVICE float3x3 & multMatTAdd(float3x3 const & B, float3x3 & result) const
	{
		for(int i=0; i<3; ++i)
			for(int j=0; j<3; ++j){
				float sum(0);
				for(int k=0; k<3; k++)
					sum += val[i*3 + k] * B[j*3 + k];
				result[i*3 + j] += sum;
				}
		return result;
	}


	/// Product of matrix and vector b. Result is written to result and its reference is returned.
	CUDA_VEC_HOSTDEVICE float3 & multVec(float3 const & b, float3 & result) const
	{
        return result = make_float3(
            val[0] * b.x + val[1] * b.y + val[2] * b.z,
            val[3] * b.x + val[4] * b.y + val[5] * b.z,
            val[6] * b.x + val[7] * b.y + val[8] * b.z);
	}


	/// Product of matrix and vector b. Result is added to vector result and its reference is returned.
	CUDA_VEC_HOSTDEVICE float3 & multVecAdd(float3 const & b, float3 & result) const
	{
        return result = make_float3(
            result.x + val[0] * b.x + val[1] * b.y + val[2] * b.z,
            result.y + val[3] * b.x + val[4] * b.y + val[5] * b.z,
            result.z + val[6] * b.x + val[7] * b.y + val[8] * b.z);
	}


	/// Product of transposed matrix and vector b. Result is written to result and its reference is returned.
	CUDA_VEC_HOSTDEVICE float3 & multTVec(float3 const & b, float3 & result) const
	{
        return result = make_float3(
            val[0] * b.x + val[3] * b.y + val[6] * b.z,
            val[1] * b.x + val[4] * b.y + val[7] * b.z,
            val[2] * b.x + val[5] * b.y + val[8] * b.z);
	}


	/// Product of transposed matrix and vector b. Result is added to vector result and its reference is returned.
	CUDA_VEC_HOSTDEVICE float3 & multTVecAdd(float3 const & b, float3 & result) const
	{
        return result = make_float3(
            result.x + val[0] * b.x + val[3] * b.y + val[6] * b.z,
            result.y + val[1] * b.x + val[4] * b.y + val[7] * b.z,
            result.z + val[2] * b.x + val[5] * b.y + val[8] * b.z);
	}

	/// Transpose matrix. Result is written to result and its reference is returned.
	CUDA_VEC_HOSTDEVICE float3x3 & transpose(float3x3 & result) const
	{
		for(int i=0; i<3; i++)
			for(int j=0; j<3; j++)
				result[3*j+i] = val[3*i+j];
		return result;
	}

	/// Gauss Elimination: Perform Gaussian elimination on matrix and given vector b. Solution is written to x.
	/** WARNING: this matrix and vector b are destroyed
 	*/
	CUDA_VEC_HOSTDEVICE void gaussElim(float3 & b, float3 & x)
	{
		for(int k=0; k<3; ++k){
			float *row = &val[k*3];
			float fac = (row[k] != float(0)) ? float(1)/row[k] : float(1);
			for(int l=k+1; l<3; ++l){
				float *actRow = &val[l*3];
				float actFac = fac * actRow[k];
				for(int ri=k+1; ri<3; ++ri)
					actRow[ri] -= actFac * row[ri];
				(&b.x)[l] -= actFac * (&b.x)[k];
				}
			}

		// Back substitution
		for(int k=3-1; k>=0; --k){
			(&x.x)[k] = (&b.x)[k];
			for(int l=k+1; l<3; ++l)
				(&x.x)[k] -= (&x.x)[l] * val[k*3+l];
			(&x.x)[k] = (val[k*3+k] != float(0)) ? (&x.x)[k] / val[k*3+k] : float(0) ;
			}
	}


	/// Gauss Elimination: Perform Gaussian elimination on matrix and given vector b using row pivoting. Solution is written to x.
	/** WARNING: this matrix and vector b are destroyed
 	*/
	CUDA_VEC_HOSTDEVICE void gaussElimRowPivot(float3 & b, float3 & x)
	{
		//float3x3 A(*this);
		//float3 ba(b);

		int perm[3]; // Store row permutation
		for(int i=0; i<3; ++i)
			perm[i] = i;
		for(int k=0; k<3; ++k){

			// Find pivot
			float mx(0);
			int pi(k); // Default initilization: use current row
			for(int li=k; li<3; ++li){
				float m = fabs(val[perm[li]*3+k]);
				if (m > mx){ mx = m; pi = li;}
				}

			int old = perm[k];
			perm[k] = perm[pi]; // Update permutation
			perm[pi] = old;


			pi = perm[k]; // pi is row index of current pivot row
			float *row = &val[pi*3];
			float fac = (row[k] != float(0)) ? float(1)/row[k] : float(1);
			for(int l=k+1; l<3; ++l){
				float *actRow = &val[perm[l]*3];
				float actFac = fac * actRow[k];
				for(int ri=k+1; ri<3; ++ri)
					actRow[ri] -= actFac * row[ri];
				(&b.x)[perm[l]] -= actFac * (&b.x)[pi];
				}


			}


		// Back substitution
		for(int k=3-1; k>=0; --k){
			(&x.x)[k] = (&b.x)[perm[k]];
			for(int l=k+1; l<3; ++l)
				(&x.x)[k] -= (&x.x)[l] * val[perm[k]*3+l];
			(&x.x)[k] = (val[perm[k]*3+k] != float(0)) ? (&x.x)[k] / val[perm[k]*3+k] : float(1) ;
			}
	}


	/// Compute largest eigenvalue (in lambda) and eigenvector (in x). Perform power method. Returns convergence value, that should be close to 1 (positive eigenvalue) or -1 (negative eigenvalue). Optional arguments are the maximum number of iteration steps, and a flag indicates whether x is used as start vector, or a start vector is computed from the matrix.
	CUDA_VEC_HOSTDEVICE float largestEigenvec(float & lambda, float3 & x, int maxSteps=25, bool computeStartVec=true)
	{
		float err(0);
		float3 xold;
		if (computeStartVec)
		{
			float mx(0);
			int iMax=0;
			for(int i=0; i<3; ++i)
			{
				lambda = norm(getRow(i));
				if (lambda > mx){ mx = lambda; iMax = i; }
			}
			x = getRow(iMax);
		}
		// Normalize x
		lambda = norm(x);
		x *= (lambda != float(0.0)) ? float(1.0)/lambda : float(1.0);
		int ic(0);
		do{
			xold = x;
			multVec(xold, x);

			lambda = norm(x);
			x *= (lambda != float(0.0)) ? float(1.0)/lambda : float(1.0);

			err = dot(x, xold);

			++ic;
		}
		while( (err*err < float(0.98)) && (ic <= maxSteps) );

		if (err < float(0)) lambda = -lambda;
		return err;

	}


	CUDA_VEC_HOSTDEVICE float largestEigenvec2(float & lambda, float3 & x, int maxSteps=25, bool computeStartVec=true)
	{
		float err(0);
		float fnorm;
		float3 xold;
		if (computeStartVec)
		{
			float mx(0);
			int iMax=0;
			for(int i=0; i<3; ++i)
			{
				lambda = norm(getRow(i));
				if (lambda > mx){ mx = lambda; iMax = i; }
			}
			x = getRow(iMax);
		}

		fnorm = norm(x);
		x *= (fnorm != float(0.0)) ? float(1.0)/fnorm : float(1.0);

		int ic(0);
		do{
			multVec(x, xold);
			lambda =  dot(x, xold) / norm_square(x);
			fnorm = norm(xold);
			xold *= (fnorm != float(0.0)) ? float(1.0)/fnorm : float(1.0);
			err = dot(x, xold);
			if (err*err > float(0.98)) break;

			float3x3 tmp(*this);
			for(int i=0; i<3; i++)
				tmp.get(i,i) -= lambda;

			tmp.gaussElim(x, xold);
			x = xold;
			fnorm = norm(x);
			x *= (fnorm != float(0.0)) ? float(1.0)/fnorm : float(1.0);

			++ic;
		}
		while( (err*err < float(0.98)) && (ic <= maxSteps) );

		if (err < float(0)) lambda = -lambda;
		return err;

	}


	// @}
};

CUDA_VEC_HOSTDEVICE float3 operator* (float3 const & v, float3x3 const & m) {
    float3 result;
    m.multTVec(v, result);
    return result;
}

// float4x4 class
////////////////////////////////////////////////////////////////////////////////

class float4x4
{
	float val[4*4];

public:

#ifndef __CUDACC__ // CUDA doesn't support non-empty constructors

	/// @name Constructors
	// @{
	/// Default constructor
	float4x4()
	{ for(int k=0; k<4*4; ++k) val[k] = 0.0f; }

	/// Copy constructor
	float4x4(float4x4 const & m)
	{
		for(int k=0; k<4*4; ++k) val[k] = m[k];
	}

	/// Copy constructor with a given array (must have a length >= 4*4)
	float4x4(float const * m)
	{
		for(int k=0; k<4*4; ++k) val[k] = m[k];
	}

	/// Constructor with an initial value for all elements
	explicit float4x4(float const & w)
	{
		for(int k=0; k<4*4; ++k) val[k] = w;
	}
	// @}

	// DirectX
	#ifdef __D3DX9MATH_H__
        __host__ float4x4( const D3DXMATRIX& other ) {
            for(int i=0; i<16; i++)
                val[i] = other[i];
        }

		D3DXMATRIX toD3DXMAT() const {return D3DXMATRIX(FLOAT(val[0]),FLOAT(val[1]),FLOAT(val[2]),FLOAT(val[3]),
														FLOAT(val[4]),FLOAT(val[5]),FLOAT(val[6]),FLOAT(val[7]),
														FLOAT(val[8]),FLOAT(val[9]),FLOAT(val[10]),FLOAT(val[11]),
														FLOAT(val[12]),FLOAT(val[13]),FLOAT(val[14]),FLOAT(val[15]));}
		operator D3DXMATRIX(void) const {return toD3DXMAT();}
	#endif

#endif

	/// @name Array subscripting, matrix access and dereferencing operators
	// @{
	/// Array subscripting operator
	CUDA_VEC_HOSTDEVICE float & operator[](int k)
	{
		//CUDA_VEC_ASSERT( (k>=0) && (k<4*4) );
		return val[k];
	}

	/// Constant array subscripting operator.
	CUDA_VEC_HOSTDEVICE const float & operator[](int k) const
	{
		//CUDA_VEC_ASSERT( (k>=0) && (k<4*4) );
		return val[k];
	}

	/// Matrix access
	CUDA_VEC_HOSTDEVICE float & get(int k, int l)
	{
		CUDA_VEC_ASSERT( (k>=0) && (l>=0) && (k<4) && (l<4) );
		return val[k*4+l];
	}

	/// Matrix access
	CUDA_VEC_HOSTDEVICE const float & get(int k, int l) const
	{
		CUDA_VEC_ASSERT( (k>=0) && (l>=0) && (k<4) && (l<4) );
		return val[k*4+l];
	}

	/// Matrix access to column vectors
	CUDA_VEC_HOSTDEVICE float4 getCol(int l) const
	{
		CUDA_VEC_ASSERT( (l>=0) && (l<4) );
		float4 v;
		for(int k=0; k<4; ++k)
			(&v.x)[k] = val[k*4+l];
		return v;
	}

	/// Matrix access to row vectors
	CUDA_VEC_HOSTDEVICE float4 getRow(int k) const
	{
		CUDA_VEC_ASSERT( (k>=0) && (k<4) );
        int r = k*4;
		return make_float4(val[r], val[r+1], val[r+2], val[r+3]);
	}


	/// Dereferencing operator
	CUDA_VEC_HOSTDEVICE operator float *()
	{
		return val;
	}

	/// Constant dereferencing operator
	CUDA_VEC_HOSTDEVICE operator const float *() const
	{
		return val;
	}
	// @}

	/// @name Assignment operator and arithmetic assignment operators
	// @{
	/// Assignmet operator
	CUDA_VEC_HOSTDEVICE float4x4 & operator=(float4x4 const & m)
	{
		for(int k=0; k<4*4; ++k) val[k] = m[k];
		return (*this);
	}

	/// Add and assign
	CUDA_VEC_HOSTDEVICE float4x4 & operator+=(float4x4 const & m)
	{
		for(int k=0; k<4*4; ++k) val[k] += m[k];
		return (*this);
	}

	/// Subtract and assign
	CUDA_VEC_HOSTDEVICE float4x4 &operator-=(const float4x4 &m)
	{
		for(int k=0; k<4*4; ++k) val[k] -= m[k];
		return (*this);
	}

	/// Multiply a scalar and assign
	CUDA_VEC_HOSTDEVICE float4x4 & operator*=(float const & w)
	{
		for(int k=0; k<4*4; ++k) val[k] *= w;
		return (*this);
	}

	/// Multiply a matrix and assign
	CUDA_VEC_HOSTDEVICE float4x4 & operator*=(float4x4 const & m)
	{
		float4x4 result;
		for(int i=0; i<4; ++i)
			for(int j=0; j<4; ++j){
				float sum(0);
				for(int k=0; k<4; k++)
					sum += val[i*4 + k] * m[k*4 + j];
				result[i*4 + j] = sum;
				}
		*this = result;
		return *this;
	}

	/// Divide by a scalar and assign
	CUDA_VEC_HOSTDEVICE float4x4 & operator/=(float const & w)
	{
		for(int k=0; k<4*4; ++k) val[k] /= w;
		return (*this);
	}

	///// Modulo by a scalar and assign
	//float4x4 &operator%=(const float &w)
	//{
	//	for(int k=0; k<4*4; ++k) val[k] %= w;
	//	return (*this);
	//}

	/// Sum of two matrices
	CUDA_VEC_HOSTDEVICE float4x4 operator+(float4x4 const & m) const
	{
		float4x4 res;
		for(int k=0; k<4*4; ++k) res[k] = val[k] + m[k];
		return res;
	}

	/// Difference of two matrices
	CUDA_VEC_HOSTDEVICE float4x4 operator-(float4x4 const & m) const
	{
		float4x4 res;
		for(int k=0; k<4*4; ++k) res[k] = val[k] - m[k];
		return res;
	}

	/// Multiply matrix by scalar
	CUDA_VEC_HOSTDEVICE float4x4 operator*(float const & w) const
	{
		float4x4 res;
		for(int k=0; k<4*4; ++k) res[k] = val[k] * w;
		return res;
	}

	/// Product of matrix and vector
	CUDA_VEC_HOSTDEVICE const float4 operator*(float4 const &  v) const
	{
        return make_float4(
            val[0] * v.x + val[1] * v.y + val[2] * v.z + val[3] * v.w,
            val[4] * v.x + val[5] * v.y + val[6] * v.z + val[7] * v.w,
            val[8] * v.x + val[9] * v.y + val[10] * v.z + val[11] * v.w,
            val[12] * v.x + val[13] * v.y + val[14] * v.z + val[15] * v.w);
	}

	/// Product of two matrices
	CUDA_VEC_HOSTDEVICE float4x4 operator*(float4x4 const & m) const
	{
		float4x4 res;
		for(int i=0; i<4; ++i)
			for(int j=0; j<4; ++j){
				float sum(0);
				for(int k=0; k<4; k++)
					sum += val[i*4 + k] * m[k*4 + j];
				res[i*4 + j] = sum;
				}
		return res;
	}

	/// Divide matrix by scalar
	CUDA_VEC_HOSTDEVICE float4x4 operator/(float w) const
	{
		float4x4 res;
		for(int k=0; k<4*4; ++k) res[k] = val[k] / w;
		return res;
	}

	/// Unary -
	CUDA_VEC_HOSTDEVICE float4x4 operator-() const
	{
		float4x4 res;
		for(int k=0; k<4*4; ++k) res[k] = -val[k];
		return res;
	}

	// @}

	/// @name Matrix functions
	// @{

	/// Clear the matrix to zero
	CUDA_VEC_HOSTDEVICE void clear()
	{
		for(int k=0; k<4*4; ++k)
			val[k] = float(0);
	}

	/// Multiply with matrix B and store result in result
	CUDA_VEC_HOSTDEVICE float4x4 & multMat(float4x4 const & B, float4x4 & result) const
	{
		for(int i=0; i<4; ++i)
			for(int j=0; j<4; ++j){
				float sum(0);
				for(int k=0; k<4; k++)
					sum += val[i*4 + k] * B[k*4 + j];
				result[i*4 + j] = sum;
				}
		return result;
	}


	/// Multiply with matrix B and accumulate result to matrix result
	CUDA_VEC_HOSTDEVICE float4x4 & multMatAdd(float4x4 const & B, float4x4 & result) const
	{
		for(int i=0; i<4; ++i)
			for(int j=0; j<4; ++j){
				float sum(0);
				for(int k=0; k<4; k++)
					sum += val[i*4 + k] * B[k*4 + j];
				result[i*4 + j] += sum;
				}
		return result;
	}


	/// Multiply with transposed matrix B and store result in result
	CUDA_VEC_HOSTDEVICE float4x4 & multMatT(float4x4 const & B, float4x4 & result) const
	{
		for(int i=0; i<4; ++i)
			for(int j=0; j<4; ++j){
				float sum(0);
				for(int k=0; k<4; k++)
					sum += val[i*4 + k] * B[j*4 + k];
				result[i*4 + j] = sum;
				}
		return result;
	}

	/// Multiply with transposed matrix B and accumulate result to matrix result
	CUDA_VEC_HOSTDEVICE float4x4 & multMatTAdd(const float4x4 &B, float4x4 &result) const
	{
		for(int i=0; i<4; ++i)
			for(int j=0; j<4; ++j){
				float sum(0);
				for(int k=0; k<4; k++)
					sum += val[i*4 + k] * B[j*4 + k];
				result[i*4 + j] += sum;
				}
		return result;
	}


	/// Product of matrix and vector b. Result is written to result and its reference is returned.
	CUDA_VEC_HOSTDEVICE float4 & multVec(float4 const & b, float4 & result) const
	{
        return result = make_float4(
            val[0] * b.x + val[1] * b.y + val[2] * b.z + val[3] * b.w,
            val[4] * b.x + val[5] * b.y + val[6] * b.z + val[7] * b.w,
            val[8] * b.x + val[9] * b.y + val[10] * b.z + val[11] * b.w,
            val[12] * b.x + val[13] * b.y + val[14] * b.z + val[15] * b.w);
	}


	/// Product of matrix and vector b. Result is added to vector result and its reference is returned.
	CUDA_VEC_HOSTDEVICE float4 & multVecAdd(float4 const & b, float4 & result) const
	{
        return result = make_float4(
            result.x + val[0] * b.x + val[1] * b.y + val[2] * b.z + val[3] * b.w,
            result.y + val[4] * b.x + val[5] * b.y + val[6] * b.z + val[7] * b.w,
            result.z + val[8] * b.x + val[9] * b.y + val[10] * b.z + val[11] * b.w,
            result.w + val[12] * b.x + val[13] * b.y + val[14] * b.z + val[15] * b.w);
	}


	/// Product of transposed matrix and vector b. Result is written to result and its reference is returned.
	CUDA_VEC_HOSTDEVICE float4 & multTVec(float4 const & b, float4 & result) const
	{
        return result = make_float4(
            val[0] * b.x + val[4] * b.y + val[8] * b.z + val[12] * b.w,
            val[1] * b.x + val[5] * b.y + val[9] * b.z + val[13] * b.w,
            val[2] * b.x + val[6] * b.y + val[10] * b.z + val[14] * b.w,
            val[3] * b.x + val[7] * b.y + val[11] * b.z + val[15] * b.w);
	}


	/// Product of transposed matrix and vector b. Result is added to vector result and its reference is returned.
	CUDA_VEC_HOSTDEVICE float4 & multTVecAdd(float4 const & b, float4 & result) const
	{
        return result = make_float4(
            result.x + val[0] * b.x + val[4] * b.y + val[8] * b.z + val[12] * b.w,
            result.y + val[1] * b.x + val[5] * b.y + val[9] * b.z + val[13] * b.w,
            result.z + val[2] * b.x + val[6] * b.y + val[10] * b.z + val[14] * b.w,
            result.w + val[3] * b.x + val[7] * b.y + val[11] * b.z + val[15] * b.w);
	}

	/// Transpose matrix. Result is written to result and its reference is returned.
	CUDA_VEC_HOSTDEVICE float4x4 & transpose(float4x4 & result) const
	{
		for(int i=0; i<4; i++)
			for(int j=0; j<4; j++)
				result[4*j+i] = val[4*i+j];
		return result;
	}

	/// Gauss Elimination: Perform Gaussian elimination on matrix and given vector b. Solution is written to x.
	/** WARNING: this matrix and vector b are destroyed
 	*/
	CUDA_VEC_HOSTDEVICE void gaussElim(float4 & b, float4 & x)
	{
		for(int k=0; k<4; ++k){
			float *row = &val[k*4];
			float fac = (row[k] != float(0)) ? float(1)/row[k] : float(1);
			for(int l=k+1; l<4; ++l){
				float *actRow = &val[l*4];
				float actFac = fac * actRow[k];
				for(int ri=k+1; ri<4; ++ri)
					actRow[ri] -= actFac * row[ri];
				(&b.x)[l] -= actFac * (&b.x)[k];
				}
			}

		// Back substitution
		for(int k=4-1; k>=0; --k){
			(&x.x)[k] = (&b.x)[k];
			for(int l=k+1; l<4; ++l)
				(&x.x)[k] -= (&x.x)[l] * val[k*4+l];
			(&x.x)[k] = (val[k*4+k] != float(0)) ? (&x.x)[k] / val[k*4+k] : float(0) ;
			}
	}


	/// Gauss Elimination: Perform Gaussian elimination on matrix and given vector b using row pivoting. Solution is written to x.
	/** WARNING: this matrix and vector b are destroyed
 	*/
	CUDA_VEC_HOSTDEVICE void gaussElimRowPivot(float4 & b, float4 & x)
	{
		//float4x4 A(*this);
		//float4 ba(b);

		int perm[4]; // Store row permutation
		for(int i=0; i<4; ++i)
			perm[i] = i;
		for(int k=0; k<4; ++k){

			// Find pivot
			float mx(0);
			int pi(k); // Default initilization: use current row
			for(int li=k; li<4; ++li){
				float m = fabs(val[perm[li]*4+k]);
				if (m > mx){ mx = m; pi = li;}
				}

			int old = perm[k];
			perm[k] = perm[pi]; // Update permutation
			perm[pi] = old;


			pi = perm[k]; // pi is row index of current pivot row
			float *row = &val[pi*4];
			float fac = (row[k] != float(0)) ? float(1)/row[k] : float(1);
			for(int l=k+1; l<4; ++l){
				float *actRow = &val[perm[l]*4];
				float actFac = fac * actRow[k];
				for(int ri=k+1; ri<4; ++ri)
					actRow[ri] -= actFac * row[ri];
				(&b.x)[perm[l]] -= actFac * (&b.x)[pi];
				}


			}


		// Back substitution
		for(int k=4-1; k>=0; --k){
			(&x.x)[k] = (&b.x)[perm[k]];
			for(int l=k+1; l<4; ++l)
				(&x.x)[k] -= (&x.x)[l] * val[perm[k]*4+l];
			(&x.x)[k] = (val[perm[k]*4+k] != float(0)) ? (&x.x)[k] / val[perm[k]*4+k] : float(1) ;
			}
	}


	/// Compute largest eigenvalue (in lambda) and eigenvector (in x). Perform power method. Returns convergence value, that should be close to 1 (positive eigenvalue) or -1 (negative eigenvalue). Optional arguments are the maximum number of iteration steps, and a flag indicates whether x is used as start vector, or a start vector is computed from the matrix.
	CUDA_VEC_HOSTDEVICE float largestEigenvec(float & lambda, float4 & x, int maxSteps=25, bool computeStartVec=true)
	{
		float err(0);
		float4 xold;
		if (computeStartVec)
		{
			float mx(0);
			int iMax=0;
			for(int i=0; i<4; ++i)
			{
				lambda = norm(getRow(i));
				if (lambda > mx){ mx = lambda; iMax = i; }
			}
			x = getRow(iMax);
		}
		// Normalize x
		lambda = norm(x);
		x *= (lambda != float(0.0)) ? float(1.0)/lambda : float(1.0);
		int ic(0);
		do{
			xold = x;
			multVec(xold, x);

			lambda = norm(x);
			x *= (lambda != float(0.0)) ? float(1.0)/lambda : float(1.0);

			err = dot(x, xold);

			++ic;
		}
		while( (err*err < float(0.98)) && (ic <= maxSteps) );

		if (err < float(0)) lambda = -lambda;
		return err;

	}


	CUDA_VEC_HOSTDEVICE float largestEigenvec2(float & lambda, float4 & x, int maxSteps=25, bool computeStartVec=true)
	{
		float err(0);
		float fnorm;
		float4 xold;
		if (computeStartVec)
		{
			float mx(0);
			int iMax=0;
			for(int i=0; i<4; ++i)
			{
				lambda = norm(getRow(i));
				if (lambda > mx){ mx = lambda; iMax = i; }
			}
			x = getRow(iMax);
		}

		fnorm = norm(x);
		x *= (fnorm != float(0.0)) ? float(1.0)/fnorm : float(1.0);

		int ic(0);
		do{
			multVec(x, xold);
			lambda =  dot(x, xold) / norm_square(x);
			fnorm = norm(xold);
			xold *= (fnorm != float(0.0)) ? float(1.0)/fnorm : float(1.0);
			err = dot(x, xold);
			if (err*err > float(0.98)) break;

			float4x4 tmp(*this);
			for(int i=0; i<4; i++)
				tmp.get(i,i) -= lambda;

			tmp.gaussElim(x, xold);
			x = xold;
			fnorm = norm(x);
			x *= (fnorm != float(0.0)) ? float(1.0)/fnorm : float(1.0);

			++ic;
		}
		while( (err*err < float(0.98)) && (ic <= maxSteps) );

		if (err < float(0)) lambda = -lambda;
		return err;

	}


	// @}
};

CUDA_VEC_HOSTDEVICE float4 operator* (float4 const & v, float4x4 const & m) {

	printf("Error");
    float4 result;
    m.multTVec(v, result);
    return result;
}

CUDA_VEC_HOSTDEVICE void print_float4x4(const float4x4 & in,
		const char* name,
		const int arg = 0)
{
	printf("%s(%i): \n %f, %f, %f, %f \n %f, %f, %f, %f \n %f, %f, %f, %f \n %f, %f, %f, %f \n", name, arg,
			in[0], in[1], in[2], in[3],
			in[4], in[5], in[6], in[7],
			in[8], in[9], in[10], in[11],
			in[12], in[13], in[14], in[15]);
}

CUDA_VEC_HOSTDEVICE void print_float3x3(const float3x3 & in,
		const char* name)
{
	printf("%s: %f %f %f \n %f %f %f \n %f %f %f \n", name, in[0], in[1], in[2],
			in[0+3], in[1+3], in[2+3],
			in[0+6], in[1+6], in[2+6]);
}

//#ifndef __CUDACC__
//inline float3x3::float3x3( const float4x4& other ) : m11(other.m11), m12(other.m12), m13(other.m13),
//        m21(other.m21), m22(other.m22), m23(other.m23),
//        m31(other.m31), m32(other.m32), m33(other.m33) {};
//#endif

#endif
