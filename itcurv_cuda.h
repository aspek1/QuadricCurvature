/*
 * itcurv_cuda.h
 *
 *  Created on: 05/06/2014
 *      Author: andrew
 */

#ifndef ITCURV_CUDA
#define ITCURV_CUDA


#include "cuda_ext_vecs.h"
#include "cuda_operators.h"
//#include "GLCudaInterop.hpp"
#include "cholesky.h"
#include "curvQE.h"

//#include "cuda_array.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

//using namespace CVD;
//using namespace TooN;

// one generator for each parameter in iterative curvature measurement
__constant__ float4x4 E_x;
__constant__ float4x4 E_y;
__constant__ float4x4 E_a;
__constant__ float4x4 Q_x;
__constant__ float4x4 Q_y;
__constant__ float4x4 Q_xy;

__constant__ float4x4 E_x_T;
__constant__ float4x4 E_y_T;
__constant__ float4x4 E_a_T;
__constant__ float4x4 Q_x_T;
__constant__ float4x4 Q_y_T;
__constant__ float4x4 Q_xy_T;


__device__ void print_float4(const float4 & in,
		const char* name)
{
	printf("%s: \n %f, %f, %f, %f \n", name, in.x, in.y, in.z, in.w);
}




void update_curvature_gaussian(const int radius, const float sigma)
{
	int size = (radius*2+1)*(radius*2+1);
	float* h_kernel = new float[size];

	for(int i = 0; i<size; ++i){
		float x_diff = (i%(2*radius+1) - radius)*2;
		float y_diff = (i/(2*radius+1) - radius)*2;
		h_kernel[i] = exp(-(x_diff*x_diff + y_diff*y_diff)/(2*sigma));
//		std::cout << "i=" << i << ": " << h_kernel[i]<<std::endl;
	}
	cudaMemcpyToSymbol(gaussian_weights, h_kernel, sizeof(float)*size);

	delete[] h_kernel;
}

void update_weighting_constant(float new_weight)
{
	cudaMemcpyToSymbol(weight_constant, &new_weight, sizeof(float));
}

__device__ void get_rotation_matrix_from_vector(const float3 & w,
		float3x3 * result)

{
	const float theta_sq = dot(w,w);
	const float theta = sqrt(theta_sq);

	const float one_6th = 1.0f/6.0f;
	const float one_20th = 1.0f/20.0f;
	float A, B;
	//Use a Taylor series expansion near zero. This is required for
	//accuracy, since sin t / t and (1-cos t)/t^2 are both 0/0.
	if (theta_sq < 1e-8) {
		A = 1.0 - one_6th * theta_sq;
		B = 0.5;
	} else {
		if (theta_sq < 1e-6) {
			B = 0.5 - 0.25 * one_6th * theta_sq;
			A = 1.0 - theta_sq * one_6th*(1.0 - one_20th * theta_sq);
		} else {
			const float inv_theta = 1.0/theta;
			A = sin(theta) * inv_theta;
			B = (1 - cos(theta)) * (inv_theta * inv_theta);
		}
	}

	//rodrigues_so3_exp(w, A, B, result.my_matrix);
	//const Vector<S,VP, VA>& w, const float A, const float B, Matrix<3,3,float,MA>& R){

	{
		const float wx2 = (float)w.x*w.x;
		const float wy2 = (float)w.y*w.y;
		const float wz2 = (float)w.z*w.z;

		(*result)[0] = 1.0 - B*(wy2 + wz2);
		(*result)[4] = 1.0 - B*(wx2 + wz2);
		(*result)[8] = 1.0 - B*(wx2 + wy2);
	}
	{
		const float a = A*w.z;
		const float b = B*(w.x*w.y);
		(*result)[1] = b - a;
		(*result)[3] = b + a;
	}
	{
		const float a = A*w.y;
		const float b = B*(w.x*w.z);
		(*result)[2] = b + a;
		(*result)[6] = b - a;
	}
	{
		const float a = A*w.x;
		const float b = B*(w.y*w.z);
		(*result)[5] = b - a;
		(*result)[7] = b + a;
	}

}

__device__ void get_vector_from_rotation_matrix(const float3x3 mat,
		float3 & vect)
{
	const float cos_angle = (mat[0] + mat[4] + mat[8] - 1.0) * 0.5;
	vect.x = (mat[7]-mat[5])/2;
	vect.y = (mat[2]-mat[6])/2;
	vect.z = (mat[3]-mat[1])/2;

	float sin_angle_abs = sqrt(dot(vect,vect));
	if (cos_angle > M_SQRT1_2) {            // [0 - Pi/4[ use asin
		if(sin_angle_abs > 0){
			vect *= asin(sin_angle_abs) / sin_angle_abs;
		}
	} else if( cos_angle > -M_SQRT1_2) {    // [Pi/4 - 3Pi/4[ use acos, but antisymmetric part
		const float angle = acos(cos_angle);
		vect *= angle / sin_angle_abs;
	} else {  // rest use symmetric part
		// antisymmetric part vanishes, but still large rotation, need information from symmetric part
		const float angle = M_PI - asin(sin_angle_abs);
		const float d0 = mat[0] - cos_angle,
			d1 = mat[4] - cos_angle,
			d2 = mat[8] - cos_angle;

		float3 r2;

		if(d0*d0 > d1*d1 && d0*d0 > d2*d2)
			r2 = make_float3(d0, (mat[3]+mat[1])/2, (mat[2]+mat[6])/2);
		else if(d1*d1 > d2*d2)
			r2 = make_float3((mat[3]+mat[1])/2, d1, (mat[7]+mat[5])/2);
		else
			r2 = make_float3((mat[2]+mat[6])/2, r2.y = (mat[7]+mat[5])/2, r2.z = d2);

		// flip, if we point in the wrong direction!
		if(dot(r2, vect) < 0)
			r2 *= -1;
		r2 = normalize(r2);
		vect = angle*r2;
	}
}

__device__ void update_rotation(const float3 & w,
		float3x3 & result)
{
	float3x3 delta_R;
	get_rotation_matrix_from_vector(w, &delta_R);
	result = delta_R*result;
}

__device__ float get_total_error(const float4* coords,
		const float4 & center,
		const int center_idx,
		curvQE & QE,
		const int length_offsets,
		const float3 stats)
{
	float total_error = 0.0f;

	for(int offset_idx = 0; offset_idx<length_offsets; ++offset_idx)
	{
		int shifted_idx = center_idx + index_offsets[offset_idx];

		if(coords[shifted_idx].z == 0) continue;

		float4 transformed_coord = coords[shifted_idx] - center;
		transformed_coord.w = 1.0f; // remain homogeneous;

		float err = QE.get_error(transformed_coord);
		float weight = fabs(stats.x) / (fabs(stats.x) + fabs(err));

		total_error += weight*fabs(err);
	}

	return total_error;
}
#include <math.h>

__device__ float3 get_error_stats(curvQE &QE,
		const float4* coords,
		const float4 &center_coord,
		const int &img_size,
		const int &idx,
		const int &num_offsets)
{
	float total_err = 0.0f;
	float total_sqr_err = 0.0f;
	int count = 0;
	for(int offset_idx = 0; offset_idx<num_offsets; ++offset_idx)
	{
		int shifted_idx = idx + index_offsets[offset_idx];
		if(shifted_idx < 0 || shifted_idx >= img_size || coords[shifted_idx].z == 0) continue;
		count++;

		float4 transformed_coord = coords[shifted_idx] - center_coord;
		transformed_coord.w = 1.0f;

		float error = QE.get_error(transformed_coord);
		total_err += error;

		float errsqr = error*error;
		total_sqr_err += errsqr;
	}

	float ave_sqr_err = total_sqr_err/(float)count;
	float ave_err = total_err/(float)count;
	float var = ave_sqr_err - ave_err*ave_err;

	return make_float3(ave_err, ave_sqr_err, var);
}

__global__ void compute_curvature_PCL(const float4* coords,
		float3* normals,
		float3* curvatures,
		const int width,
		const int height)
{
	int xIdx = threadIdx.x + blockIdx.x*blockDim.x;
	int yIdx = threadIdx.y + blockIdx.y*blockDim.y;

	if(xIdx >= width || yIdx >= height) return; // borders


}

/// Computing iterative curvature estimate
__global__ void compute_iterative_curvature(const float4* coords,
		float3* normals,
		float3* curvatures,
		const int width,
		const int height,
		const int kernel_radius,
		const int max_iteration,
		float4* errors)
{
	int xIdx = threadIdx.x + blockIdx.x*blockDim.x;
	int yIdx = threadIdx.y + blockIdx.y*blockDim.y;

	if(xIdx >= width || yIdx >= height) return; // borders

	__shared__ extern float shared_array[];

	int idx = toIndex(xIdx, yIdx, width);

	/// initialize the central point and normal
	const float3 c_norm = normals[idx];

	// bail out early if the central point or normal is bad
	if(coords[idx].z == 0 || dot(c_norm, c_norm) < .9) {
		curvatures[idx] = make_float3(0,0,-1);
		return;
	}

	wls_LVM wls_lvm(width, height, idx, coords[idx], max_iteration, 1e-4f);

	// initialize frame for E
	wls_lvm.init_QE(c_norm);

	// Initialise shared memory object
	int shared_idx = (threadIdx.x + blockDim.x*threadIdx.y)*6;
	float* J = &shared_array[shared_idx];

	wls_lvm.compute_curvature(coords, J);

	float3 new_norm = make_float3(wls_lvm.get_QE()->get_rot_mat()._row2);// normalize(se3->rotate(normals[idx]));
	if(!isnan(new_norm.x) && !isnan(new_norm.y) && !isnan(new_norm.z))
		normals[idx] = new_norm; //normalize(se3->rotate(normals[idx]));
//
//	SE3 * se3 = wls_lvm.get_norm_update();
//	float3 new_norm = normalize(se3->rotate(normals[idx]));
//	if(!isnan(new_norm.x) && !isnan(new_norm.y) && !isnan(new_norm.z))
//		normals[idx] = normalize(se3->rotate(normals[idx]));

	curvQE* QE = wls_lvm.get_QE();
	float2 ks = QE->get_principal_curvatures();
//	if((ks.x == 0 and ks.y == 0))
//		printf("Zero at: %d", idx);
	if(isnan(ks.x) or isnan(ks.y))
		curvatures[idx] = make_float3(0, 0, -1);
	else{
		float max = 0.3;
		ks.x = abs(ks.x) > max ? (ks.x < 0 ? -max : max) : ks.x;
		ks.y = abs(ks.y) > max ? (ks.y < 0 ? -max : max) : ks.y;
		curvatures[idx] = make_float3( ks.x, ks.y, 0);

	}
	/// Multiple orders of magnitude difference between mean ((k1+k2)/2) and gaussian (k1*k2)
	//(ks.x+ks.y)/2, ks.x*ks.y, 0);
	return;
}


__global__ void compute_iterative_quadrics(const float4* coords,
		float3* normals,
		float3* curvatures,
		float* quadrics,
		const int width,
		const int height,
		const int kernel_radius,
		const int max_iteration,
		float4* errors)
{
	int xIdx = threadIdx.x + blockIdx.x*blockDim.x;
	int yIdx = threadIdx.y + blockIdx.y*blockDim.y;

	if(xIdx >= width || yIdx >= height) return; // borders

	__shared__ extern float shared_array[];

	int idx = toIndex(xIdx, yIdx, width);

	/// initialize the central point and normal
	const float3 c_norm = normals[idx];

	// bail out early if the central point or normal is bad

	if(coords[idx].z == 0  || dot(c_norm, c_norm) < .9) {
		curvatures[idx] = make_float3(0,0,-1);
		return;
	}

	wls_LVM wls_lvm(width, height, idx, coords[idx], max_iteration, 1e-4f);

	// initialize frame for E
	wls_lvm.init_QE(c_norm);

	// Initialise shared memory object
	int shared_idx = (threadIdx.x + blockDim.x*threadIdx.y)*6;
	float* J = &shared_array[shared_idx];

	wls_lvm.compute_curvature(coords, J);

//	SE3 * se3 = wls_lvm.get_norm_update();
//	float3 new_norm = normalize(se3->rotate(normals[idx]));
//	if(!isnan(new_norm.x) && !isnan(new_norm.y) && !isnan(new_norm.z))
//		normals[idx] = normalize(se3->rotate(normals[idx]));

	normals[idx] = wls_lvm.get_normal();

	curvQE* QE = wls_lvm.get_QE();
	float2 ks = QE->get_principal_curvatures();

	SE3 rotation_mat = QE->get_rot_mat();
	float3 quadric_values = QE->get_curvatures();
	float z_trans = QE->get_z_trans();



//	if((ks.x == 0 and ks.y == 0))
//		printf("Zero at: %d", idx);
	if(isnan(ks.x) or isnan(ks.y)) {
		curvatures[idx] = make_float3(0, 0, -1);


//		quadrics[idx*13] = rotation_mat._row0.x;
//		quadrics[idx*13+1] = rotation_mat._row0.y;
//		quadrics[idx*13+2] = rotation_mat._row0.z;
//		quadrics[idx*13+3] = rotation_mat._row1.x;
//		quadrics[idx*13+4] = rotation_mat._row1.y;
//		quadrics[idx*13+5] = rotation_mat._row1.z;
//		quadrics[idx*13+6] = rotation_mat._row2.x;
//		quadrics[idx*13+7] = rotation_mat._row2.y;
//		quadrics[idx*13+8] = rotation_mat._row2.z;

		quadrics[idx*13] = 0;
		quadrics[idx*13+1] = 0;
		quadrics[idx*13+2] = 0;
		quadrics[idx*13+3] = 0;
		quadrics[idx*13+4] = 0;
		quadrics[idx*13+5] = 0;
		quadrics[idx*13+6] = 0;
		quadrics[idx*13+7] = 0;
		quadrics[idx*13+8] = 0;

		quadrics[idx*13+9] = 0;

		quadrics[idx*13+10] = 0;
		quadrics[idx*13+11] = 0;
		quadrics[idx*13+12] = 0;
	}
	else{
//		float max = 0.3;
//		ks.x = abs(ks.x) > max ? (ks.x < 0 ? -max : max) : ks.x;
//		ks.y = abs(ks.y) > max ? (ks.y < 0 ? -max : max) : ks.y;
		curvatures[idx] = make_float3( ks.x, ks.y, 0);


		quadrics[idx*13] = rotation_mat._row0.x;
		quadrics[idx*13+1] = rotation_mat._row0.y;
		quadrics[idx*13+2] = rotation_mat._row0.z;
		quadrics[idx*13+3] = rotation_mat._row1.x;
		quadrics[idx*13+4] = rotation_mat._row1.y;
		quadrics[idx*13+5] = rotation_mat._row1.z;
		quadrics[idx*13+6] = rotation_mat._row2.x;
		quadrics[idx*13+7] = rotation_mat._row2.y;
		quadrics[idx*13+8] = rotation_mat._row2.z;

		quadrics[idx*13+9] = z_trans;

		quadrics[idx*13+10] = quadric_values.x;
		quadrics[idx*13+11] = quadric_values.y;
		quadrics[idx*13+12] = quadric_values.z;

	}
	/// Multiple orders of magnitude difference between mean ((k1+k2)/2) and gaussian (k1*k2)
	//(ks.x+ks.y)/2, ks.x*ks.y, 0);
	return;
}


void compute_iterative_curvature_host(KeyFrame * keyframe,
		float3* curvatures,
		const int kernel_radius,
		const int max_iteration,
		float4* errors)
{
	int width = keyframe->get_width();
	int height = keyframe->get_height();

	dim3 block(16, 8, 1);
	dim3 grid(width/block.x + 1, height/block.y + 1, 1);
//	dim3 grid(10, 20, 1);
	size_t shared = sizeof(float)*block.x*block.y*6;

	compute_iterative_curvature<<<grid, block, shared>>> (keyframe->get_mapped_coords(),
			keyframe->get_mapped_normals(),
			curvatures,
			keyframe->get_width(),
			keyframe->get_height(),
			kernel_radius,
			max_iteration,
			errors);
}

void compute_quadrics_host(KeyFrame * keyframe,
		float3* curvatures,
		float* quadrics,
		const int kernel_radius,
		const int max_iteration,
		float4* errors)
{
	int width = keyframe->get_width();
	int height = keyframe->get_height();

	dim3 block(16, 8, 1);
	dim3 grid(width/block.x + 1, height/block.y + 1, 1);
//	dim3 grid(10, 20, 1);
	size_t shared = sizeof(float)*block.x*block.y*6;

	compute_iterative_quadrics<<<grid, block, shared>>> (keyframe->get_mapped_coords(),
			keyframe->get_mapped_normals(),
			curvatures,
			quadrics,
			keyframe->get_width(),
			keyframe->get_height(),
			kernel_radius,
			max_iteration,
			errors);
}

void compute_iterative_curvature_host(float4* coords,
		float3* norms,
		float3* curvatures,
		const int width,
		const int height,
		const int kernel_radius,
		const int max_iteration,
		float4* errors)
{

	dim3 block(16, 8, 1);
	dim3 grid(width/block.x + 1, height/block.y + 1, 1);
//	dim3 grid(10, 20, 1);
	size_t shared = sizeof(float)*block.x*block.y*6;

	compute_iterative_curvature<<<grid, block, shared>>> (coords,
			norms,
			curvatures,
			width,
			height,
			kernel_radius,
			max_iteration,
			errors);
}


void compute_quadrics_host(float4* coords,
		float3* norms,
		float3* curvatures,
		float* quadrics,
		const int width,
		const int height,
		const int kernel_radius,
		const int max_iteration,
		float4* errors)
{

	dim3 block(16, 8, 1);
	dim3 grid(width/block.x + 1, height/block.y + 1, 1);
//	dim3 grid(10, 20, 1);
	size_t shared = sizeof(float)*block.x*block.y*6;

	compute_iterative_quadrics<<<grid, block, shared>>> (coords,
			norms,
			curvatures,
			quadrics,
			width,
			height,
			kernel_radius,
			max_iteration,
			errors);
}



// Testing Curvature Calculation

#endif /* ITCURV_CUDA */
