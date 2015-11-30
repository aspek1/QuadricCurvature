#ifndef _CUDA_TEST
#define _CUDA_TEST

#undef PRINT_DEBUG
//#define USE_KINECT

#include <stdlib.h>
#include <stdio.h>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/glx.h>
#include <GL/glu.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cublas_v2.h>

#include <curand_kernel.h>

#include "cuda_operators.h"
#include "cuda_ext_vecs.h"
#include "itcurv_cuda.h"

//#include "my_cuda_helpers.h"
#include "GLCudaInterop.hpp"
#include "cuda_device_functions.h"

// CUDA helper functions
#define ARRAY_SIZE 100
//#define DEPTH_THRES 300 //in mm


__global__ void convertArrayToCoords(const ushort* d_array, float4* d_coords, int width, int height);
__global__ void bilateral_filter(const ushort* d_input, ushort* d_output, int r, float sigmaR, int width, int height);

__global__ void compute_normal_field(const float4* d_coords, float3* d_normals, int width, int height);
__global__ void compute_normal_field_w_smoothing(const float4* d_coords, float3* d_normals, int width, int height);

void perform_bilateral_filter(const ushort* d_input, ushort* d_output, int r, float sigmaR, int width, int height);
void convert_depth_to_xyz(ushort* d_depth, float4* d_coords, int width, int height);
void compute_normals(KeyFrame * kf);

inline __device__ float3 toFloat3(const float4 vect)
{
	return make_float3(vect.x, vect.y, vect.z);
}

/// Works for input data - mask is used to store valid depth coords
__global__ void convertArrayToCoords(const ushort* d_array, float4* d_coords, int width, int height)
{

	const int xIdx = threadIdx.x + blockDim.x*blockIdx.x;
	const int yIdx = threadIdx.y + blockDim.y*blockIdx.y;

	if(xIdx >= width || yIdx >= height)	return;

	int idx = xIdx + width*yIdx;


	if(d_array[idx] <= DEPTH_THRES) {
		d_coords[idx] = make_float4(0.0f,0.0f,0.0f,-1.0f);
		return;
	}

	float depth = (float)d_array[idx];//*0.001f; // conversion to meters

//	depth *= 0.2f; // adjustment for TUM dataset

	d_coords[idx] = toWorldCoords(xIdx, yIdx, depth);

}



__global__ void compute_normal_field(const float4* d_coords, float3* d_normals, int width, int height)
{
	const int xIdx = threadIdx.x + blockDim.x*blockIdx.x;
	const int yIdx = threadIdx.y + blockDim.y*blockIdx.y;

	if(xIdx >= width || yIdx >= height) return;

	int idx = xIdx + width*yIdx;

	if(yIdx == 0 || xIdx == width-1 || xIdx == 0 || yIdx == height-1) {
		d_normals[idx] = make_float3(0.0f,0.0f,0.0f);
		return;
	}
	int idx_up = idx - width;
	int idx_right = idx + 1;

	if(d_coords[idx].w < 0 || d_coords[idx_up].w < 0 || d_coords[idx_right].w < 0)
	{
		d_normals[idx] = make_float3(0,0,0);
		return;
	}

	float3 coord = toFloat3(d_coords[idx]);
	float3 up = toFloat3(d_coords[idx_up]);
	float3 right = toFloat3(d_coords[idx_right]);

	float3 up_vect = up - coord;
	float3 right_vect = right - coord;

	d_normals[idx] = normalize(cross(up_vect, right_vect));

}

__device__ bool valid_coord(const float4 &coord)
{
	return (coord.w != -1 and coord.z > 0);
}

__global__ void compute_normals_from_covariance(const float4* d_coords, float3* d_normals, int radius, int width, int height)
{
	const int xIdx = threadIdx.x + blockDim.x*blockIdx.x;
	const int yIdx = threadIdx.y + blockDim.y*blockIdx.y;

	if(xIdx >= width || yIdx >= height) return;

	int idx = xIdx + width*yIdx;

	if(yIdx < radius || yIdx >= height-radius || xIdx < radius || xIdx >= width-radius) {
		d_normals[idx] = make_float3(0.0f,0.0f,0.0f);
		return;
	}

	float x_mean = 0.0f;
	float y_mean = 0.0f;
	float z_mean = 0.0f;
	int num = 0;
	float4 t;
	for(int x = -radius; x<=radius; ++x) {
		for(int y = -radius; y<=radius; ++y) {
			int shifted_idx = idx + x + width*y;
			if(!valid_coord(d_coords[shifted_idx])) continue;
			t = d_coords[shifted_idx];
			x_mean += t.x;
			y_mean += t.y;
			z_mean += t.z;
			num++;
		}
	}

	float inv_num = 1.0f/(float)num;
	x_mean*=inv_num;
	y_mean*=inv_num;
	z_mean*=inv_num;

	float sum_zx = 0.0;
	float sum_zy = 0.0;

	float sum_xx = 0.0;
	float sum_xy = 0.0;
	float sum_yy = 0.0;

	for(int x = -radius; x<=radius; ++x) {
		for(int y = -radius; y<=radius; ++y) {
			int shifted_idx = idx + x + width*y;
			if(!valid_coord(d_coords[shifted_idx])) continue;
			t = d_coords[shifted_idx];
			sum_zx += (t.z-z_mean)*(t.x-x_mean);
			sum_zy += (t.z-z_mean)*(t.y-y_mean);

			sum_xx += (t.x-x_mean)*(t.x-x_mean);
			sum_xy += (t.y-y_mean)*(t.x-x_mean);
			sum_yy += (t.y-y_mean)*(t.y-y_mean);
		}
	}

	float M = sum_xx*sum_yy-sum_xy*sum_xy;

	float alpha_ = sum_yy*sum_zx - sum_xy*sum_zy;
	float beta_ = sum_xx*sum_zy - sum_xy*sum_zx;

	float3 n = make_float3(0,0,0);


	n = normalize(make_float3(-alpha_, -beta_, M));

	if(isnan(n.x) or isnan(n.y) or isnan(n.z)) d_normals[idx] = make_float3(0,0,0);
	else d_normals[idx] = n;
}

__global__ void compute_normals_from_covariance_inv_depth(const float4* d_coords, float3* d_normals, int radius, int width, int height)
{
	const int xIdx = threadIdx.x + blockDim.x*blockIdx.x;
	const int yIdx = threadIdx.y + blockDim.y*blockIdx.y;

	if(xIdx >= width || yIdx >= height) return;

	int idx = xIdx + width*yIdx;

	if(yIdx < radius || yIdx >= height-radius || xIdx < radius || xIdx >= width-radius) {
		d_normals[idx] = make_float3(0.0f,0.0f,0.0f);
		return;
	}

	float x_mean = 0.0f;
	float y_mean = 0.0f;
	float z_mean = 0.0f;
	int num = 0;
	float4 t;
	for(int x = -radius; x<=radius; ++x) {
		for(int y = -radius; y<=radius; ++y) {
			int shifted_idx = idx + x + width*y;
			if(!valid_coord(d_coords[shifted_idx])) continue;
			t = d_coords[shifted_idx];

			num++;

			float u = t.x/t.z;
			float v = t.y/t.z;
			float q = 1.0f/t.z;

			x_mean += u;
			y_mean += v;
			z_mean += q;
		}
	}

	float inv_num = 1.0f/(float)num;
	x_mean*=inv_num;
	y_mean*=inv_num;
	z_mean*=inv_num;

	float sum_zx = 0.0;
	float sum_zy = 0.0;

	float sum_xx = 0.0;
	float sum_xy = 0.0;
	float sum_yy = 0.0;

	for(int x = -radius; x<=radius; ++x) {
		for(int y = -radius; y<=radius; ++y) {
			int shifted_idx = idx + x + width*y;
			if(!valid_coord(d_coords[shifted_idx])) continue;
			t = d_coords[shifted_idx];


			float u = t.x/t.z;
			float v = t.y/t.z;
			float q = 1.0f/t.z;

			sum_zx += (q-z_mean)*(u-x_mean);
			sum_zy += (q-z_mean)*(v-y_mean);

			sum_xx += (u-x_mean)*(u-x_mean);
			sum_xy += (u-x_mean)*(v-y_mean);
			sum_yy += (v-y_mean)*(v-y_mean);
		}
	}

	float M = sum_xx*sum_yy-sum_xy*sum_xy;

	float alpha_ = sum_yy*sum_zx - sum_xy*sum_zy;
	float beta_ = sum_xx*sum_zy - sum_xy*sum_zx;

	float3 n = make_float3(0,0,0);


	n = normalize(make_float3(-alpha_, -beta_, M));

	float C = n.x*x_mean + n.y*y_mean + n.z*z_mean;

	n = normalize(make_float3(-n.x, -n.y, C)); // convert back to xyz norm

	if(isnan(n.x) or isnan(n.y) or isnan(n.z)) d_normals[idx] = make_float3(0,0,0);
	else d_normals[idx] = n;
}
__global__ void compute_normal_field_w_smoothing(const float4* d_coords, float3* d_normals, int width, int height)
{
	const int xIdx = threadIdx.x + blockDim.x*blockIdx.x;
	const int yIdx = threadIdx.y + blockDim.y*blockIdx.y;

	if(xIdx >= width || yIdx >= height) return;

	int idx = xIdx + width*yIdx;

	if(yIdx == 0 || yIdx == height-1 || xIdx == 0 || xIdx == width-1) {
		d_normals[idx] = make_float3(0.0f,0.0f,0.0f);
		return;
	}

	/// + with pos in the center
	int idx_top = idx - width;
	int idx_bottom = idx + width;
	int idx_right = idx + 1;
	int idx_left = idx - 1;

	/// X with pos in the center
	int idx_top_left = idx_top - 1;
	int idx_top_right = idx_top + 1;
	int idx_bottom_left = idx_bottom - 1;
	int idx_bottom_right = idx_bottom + 1;


	/// check valid coord

	if(d_coords[idx_top].w < 0 ||
	   d_coords[idx_bottom].w < 0 ||
	   d_coords[idx_left].w < 0 ||
	   d_coords[idx_right].w < 0 ||
	   d_coords[idx_top_left].w < 0 ||
	   d_coords[idx_top_right].w < 0 ||
	   d_coords[idx_bottom_left].w < 0 ||
	   d_coords[idx_bottom_right].w < 0)
	{
		d_normals[idx] = make_float3(0,0,0);
		return;
	}


	float3 top = toFloat3(d_coords[idx_top]);
	float3 bottom = toFloat3(d_coords[idx_bottom]);
	float3 left = toFloat3(d_coords[idx_left]);
	float3 right = toFloat3(d_coords[idx_right]);

	float3 top_left = toFloat3(d_coords[idx_top_left]);
	float3 top_right = toFloat3(d_coords[idx_top_right]);
	float3 bottom_left = toFloat3(d_coords[idx_bottom_left]);
	float3 bottom_right = toFloat3(d_coords[idx_bottom_right]);

	float3 bottom_top_vect = top - bottom;
	float3 left_right_vect = right - left;
	float3 cross_vect_1 = top_right - bottom_left;
	float3 cross_vect_2 = top_left - bottom_right;

	float3 v1 = cross(bottom_top_vect, left_right_vect);
	float3 v2 = cross(cross_vect_2, cross_vect_1);

	d_normals[idx] = normalize(v1+v2);

}

void updateCameraModel(float fx, float fy, float cx, float cy, float r_3, float r_4)
{

	float camParams[6] = {
			fx,
			fy,
			cx,
			cy,
			r_3,
			r_3
	};

    float invCamParams[6] = {
    		1.0f/fx,
    		1.0f/fy,
    		1.0f/cx,
    		1.0f/cy,
    		r_3 != 0 ? 1.0f/r_3 : 0.0,
    		r_3 != 0 ? 1.0f/r_4 : 0.0
	};

    cudaMemcpyToSymbol(cameraParams, camParams, sizeof(float)*6);
    cudaMemcpyToSymbol(invCameraParams, invCamParams, sizeof(float)*6);
}

//void updateCameraModel(float* camParams)
//{
//    float invCamParams[6] = {1.0f/camParams[0], 1.0f/camParams[1], 1.0f/camParams[2], 1.0f/camParams[3], 1.0f/camParams[4], 1.0f/camParams[5]};
//
//    cudaMemcpyToSymbol(cameraParams, camParams, sizeof(float)*6);
//    cudaMemcpyToSymbol(invCameraParams, invCamParams, sizeof(float)*6);
//}

void convert_depth_to_xyz(ushort* d_depth, float4* d_coords, int width, int height)
{

	dim3 block(16, 16, 1);
	dim3 grid(width/16+1, height/16+1, 1);

	/// compute coord array
	convertArrayToCoords<<<grid, block>>> (d_depth, d_coords, width, height);

}

void compute_normals(KeyFrame * kf)
{
	dim3 block(16, 16, 1);
	dim3 grid(kf->get_width()/block.x + 1, kf->get_height()/block.y + 1, 1);
	// shared ?
	if(kf->smooth_normals()) {
//		compute_normal_field_w_smoothing<<<grid, block>>> (kf->get_mapped_coords(), kf->get_mapped_normals(), kf->get_width(), kf->get_height());
		compute_normals_from_covariance_inv_depth<<<grid, block>>> (kf->get_mapped_coords(), kf->get_mapped_normals(), 3,  kf->get_width(), kf->get_height());
	}
	else
		compute_normal_field<<<grid, block>>> (kf->get_mapped_coords(), kf->get_mapped_normals(), kf->get_width(), kf->get_height());
}

void compute_normals(float4* coords, float3* norms, const int width, const int height, bool smooth_normals=true)
{
	dim3 block(16, 16, 1);
	dim3 grid(width/block.x + 1, height/block.y + 1, 1);
	// shared ?
	if(smooth_normals){
//		compute_normal_field_w_smoothing<<<grid, block>>> (coords, norms, width, height);
		compute_normals_from_covariance_inv_depth<<<grid, block>>> (coords, norms, 3, width, height);
	}
	else
		compute_normal_field<<<grid, block>>> (coords, norms, width, height);
}

#endif

