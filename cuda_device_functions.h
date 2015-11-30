/*
 * cuda_device_functions.h
 *
 *  Created on: 04/09/2014
 *      Author: andrew
 */

#ifndef CUDA_DEVICE_FUNCTIONS_H_
#define CUDA_DEVICE_FUNCTIONS_H_

#include "cuda_operators.h"
#include "cuda_ext_vecs.h"

#define DEPTH_THRES 300 //in mm

__constant__ float cameraParams[6]; // used for camera model
__constant__ float invCameraParams[6]; // used for camera model
__constant__ float cGaussian[64];   //gaussian array in device side

// Project from Image plane to a World Coords
inline __device__ float4 unproject(const float2& imgframe, const float depth) {
	float4 lastCamframe = make_float4(0,0,depth,1.0);

	lastCamframe.x = (imgframe.x - cameraParams[2]) * invCameraParams[0];
	lastCamframe.y = (imgframe.y - cameraParams[3]) * invCameraParams[1];

	float rSquared = lastCamframe.x * lastCamframe.x + lastCamframe.y * lastCamframe.y;
	float scaleFactor = 1 + (cameraParams[4] + cameraParams[5] * rSquared) * rSquared;

	lastCamframe.x *= scaleFactor * lastCamframe.z;
	lastCamframe.y *= scaleFactor * lastCamframe.z;
	return lastCamframe;
}


// Project from Image plane to a World Coords
inline __device__ float4 unproject_MATLAB(const float2& imgframe, const float depth) {
	float4 lastCamframe = make_float4(0,0,depth,1.0);

	lastCamframe.x = (imgframe.x - cameraParams[2]) * invCameraParams[0];
	lastCamframe.y = (imgframe.y - cameraParams[3]) * invCameraParams[1];

	float rSquared = lastCamframe.x * lastCamframe.x + lastCamframe.y * lastCamframe.y;
	float scaleFactor = 1.0/(1.0 + (cameraParams[4] + cameraParams[5] * rSquared) * rSquared);

	lastCamframe.x *= scaleFactor * depth;
	lastCamframe.y *= scaleFactor * depth;
	return lastCamframe;
}

/// Project form a Euclidean camera to image plane
inline __device__ int2 project(const float4& camframe) {

	int2 lastImgframe = make_int2(0,0);
	float inv_depth = 1.0f/camframe.z;
	float x = camframe.x*inv_depth;
	float y = camframe.y*inv_depth;
	float rSquared = x * x + y * y;
	float scaleFactor = 1 + (cameraParams[4] + cameraParams[5] * rSquared) * rSquared;
	rSquared /= scaleFactor;
	scaleFactor = 1.0f / (1 + (cameraParams[4] + cameraParams[5] * rSquared) * rSquared);

	lastImgframe.x = round(x * cameraParams[0] * scaleFactor + cameraParams[2]);
	lastImgframe.y = round(y * cameraParams[1] * scaleFactor + cameraParams[3]);
	return lastImgframe;
}

/// Project form a Euclidean camera to image plane
inline __device__ int2 project_MATLAB(const float4& camframe) {

	int2 lastImgframe = make_int2(0,0);
	float inv_depth = 1.0f/camframe.z;
	float x = camframe.x*inv_depth;
	float y = camframe.y*inv_depth;

	float rSquared = x * x + y * y;

	float scaleFactor = 1 + (cameraParams[4] + cameraParams[5] * rSquared) * rSquared;

	lastImgframe.x = round(cameraParams[0]*x*scaleFactor + cameraParams[2]);
	lastImgframe.y = round(cameraParams[1]*y*scaleFactor + cameraParams[3]);

	return lastImgframe;
}



__device__ float4 toWorldCoords(const float x, const float y, const float depth)
{
	return unproject(make_float2(x,y), depth);
}

__device__ int2 toCameraCoords(float4 coord)
{
	if(coord.z <= 1e-4) return make_int2(-1,-1); // invalid coord, check for this value
	return project(coord);
}

__device__ float getDistance(const float4 &coord1, const float4 &coord2, const float3 &normal)
{
	return dot(make_float3(coord2.x-coord1.x, coord2.y-coord1.y, coord2.z-coord1.z), normal);

//	float4 coord = coord1-coord2;
//	return coord*coord;
}



//Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float a, float b, float sigmaR)
{
    float mod = (b - a) * (b - a);
    return __expf(-mod / (2.f * sigmaR * sigmaR));
}

//column pass using coalesced global memory reads

void updateGaussian(float sigmaX, int radius)
{
    float  fGaussian[64];

    for (int i = 0; i < 2*radius + 1; ++i)
    {
        float x = i-radius;
        fGaussian[i] = expf(-(x*x) / (2*sigmaX*sigmaX));
        std::cout << "i: " << i << " " << fGaussian[i] << std::endl;
    }

    cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2*radius+1));
}


__device__ __host__ float mag(float3 f3) {
	return sqrt(f3.x*f3.x + f3.y*f3.y + f3.z*f3.z);
}


/// valid tests and invalid value specification

const float4 INVALID_DEPTH = make_float4(0,0,0,-1);
const float3 INVALID_CURVATURE = make_float3(0,0,-1);
const float3 INVALID_GRADIENT = make_float3(0,0,-1);

__device__ __host__ inline bool is_valid_depth(const float4 &val) { return !(val.w==-1); }
__device__ __host__ inline bool is_valid_curvature(const float3 &val) { return !(val.z==-1); }
__device__ __host__ inline bool is_valid_gradient(const float3 &val) { return !(val.z==-1); }



#endif /* CUDA_DEVICE_FUNCTIONS_H_ */
