/*
 * SE3.h
 *
 *  Created on: 16/09/2014
 *      Author: andrew
 */

#ifndef SE3_H_
#define SE3_H_

#define CUDA_DEVICE_HOST __device__ __host__
#include <cuda.h>
#include "cuda_ext_vecs.h"
#include "cholesky.h"
#include <math.h>
#include "cuda_device_functions.h"

//#include "GLCudaInterop.hpp"

const float one_6th = 1.0f/6.0f;
const float one_20th = 1.0f/20.0f;
const int array_size = 1024;
const float max_ave_dist = 60;

__constant__ float gaussian_weights[array_size];
__constant__ int index_offsets[array_size];
__constant__ int num_offsets;

__constant__ float weight_constant;

class SE3 {
private:

public:
	float4 _row0;
	float4 _row1;
	float4 _row2;


	CUDA_DEVICE_HOST float3 ln() {
		using std::sqrt;
		float3 result;

//		const float cos_angle = (_row0.x + my_matrix[1][1] + my_matrix[2][2] - 1.0) * 0.5;
//				result[0] = (my_matrix[2][1]-my_matrix[1][2])/2;
//				result[1] = (my_matrix[0][2]-my_matrix[2][0])/2;
//				result[2] = (my_matrix[1][0]-my_matrix[0][1])/2;

		const float cos_angle = (_row0.x + _row1.y + _row2.z - 1.0) * 0.5;
		result.x = (_row2.y-_row1.z)/2;
		result.y = (_row0.z-_row2.x)/2;
		result.z = (_row1.x-_row0.y)/2;

		float sin_angle_abs = sqrt(dot(result,result));
		if (cos_angle > M_SQRT1_2) {            // [0 - Pi/4[ use asin
			if(sin_angle_abs > 0){
				result *= asin(sin_angle_abs) / sin_angle_abs;
			}
		} else if( cos_angle > -M_SQRT1_2) {    // [Pi/4 - 3Pi/4[ use acos, but antisymmetric part
			const float angle = acos(cos_angle);
			result *= angle / sin_angle_abs;
		} else {  // rest use symmetric part
			// antisymmetric part vanishes, but still large rotation, need information from symmetric part
			const float angle = M_PI - asin(sin_angle_abs);
			const float d0 = _row0.x - cos_angle,
				d1 = _row1.y - cos_angle,
				d2 = _row2.z - cos_angle;
			float3 r2;
			if(d0*d0 > d1*d1 && d0*d0 > d2*d2){ // first is largest, fill with first column
				r2.x = d0;
				r2.y = (_row1.x+_row0.y)/2;
				r2.z = (_row0.z+_row2.x)/2;
			} else if(d1*d1 > d2*d2) { 			    // second is largest, fill with second column
				r2.x = (_row1.x+_row0.y)/2;
				r2.y = d1;
				r2.z = (_row2.y+_row1.z)/2;
			} else {							    // third is largest, fill with third column
				r2.x = (_row0.z+_row2.x)/2;
				r2.y = (_row2.y+_row1.z)/2;
				r2.z = d2;
			}
			// flip, if we point in the wrong direction!
			if(dot(r2, result) < 0)
				r2 *= -1;
			r2 = normalize(r2);
			result = angle*r2;
		}
		return result;
	}

	CUDA_DEVICE_HOST float4 mult(const float4 & vec) {
		return make_float4(dot(_row0,vec), dot(_row1,vec), dot(_row2,vec), vec.w);
	}

	CUDA_DEVICE_HOST void update_rot(float3 &w)
	{
		float3 r0 = make_float3(0,0,0);
		float3 r1 = make_float3(0,0,0);
		float3 r2 = make_float3(0,0,0);

		const float theta_sq = dot(w,w);
		const float theta = sqrt(theta_sq);

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
		const float wx2 = (float)w.x*w.x;
		const float wy2 = (float)w.y*w.y;
		const float wz2 = (float)w.z*w.z;

		r0.x = 1.0 - B*(wy2 + wz2);
		r1.y = 1.0 - B*(wx2 + wz2);
		r2.z = 1.0 - B*(wx2 + wy2);

		{
			const float a = A*w.z;
			const float b = B*(w.x*w.y);
			r0.y = b - a;
			r1.x = b + a;
		}
		{
			const float a = A*w.y;
			const float b = B*(w.x*w.z);
			r0.z = b + a;
			r2.x = b - a;
		}
		{
			const float a = A*w.x;
			const float b = B*(w.y*w.z);
			r1.z = b - a;
			r2.y = b + a;
		}

		// compute rotation change
		w = make_float3(_row0.x*r0.x + _row1.x*r0.y + _row2.x*r0.z,
				_row0.x*r1.x + _row1.x*r1.y + _row2.x*r1.z,
				_row0.x*r2.x + _row1.x*r2.y + _row2.x*r2.z);

		_row0.x = w.x;
		_row1.x = w.y;
		_row2.x = w.z;

		w = make_float3(_row0.y*r0.x + _row1.y*r0.y + _row2.y*r0.z,
				_row0.y*r1.x + _row1.y*r1.y + _row2.y*r1.z,
				_row0.y*r2.x + _row1.y*r2.y + _row2.y*r2.z);

		_row0.y = w.x;
		_row1.y = w.y;
		_row2.y = w.z;

		w = make_float3(_row0.z*r0.x + _row1.z*r0.y + _row2.z*r0.z,
				_row0.z*r1.x + _row1.z*r1.y + _row2.z*r1.z,
				_row0.z*r2.x + _row1.z*r2.y + _row2.z*r2.z);

		_row0.z = w.x;
		_row1.z = w.y;
		_row2.z = w.z;
	}

	CUDA_DEVICE_HOST float3 rotate(const float3 & vec) {
		return make_float3(vec.x*_row0.x+vec.y*_row0.y+vec.z*_row0.z,
				vec.x*_row1.x+vec.y*_row1.y+vec.z*_row1.z,
				vec.x*_row2.x+vec.y*_row2.y+vec.z*_row2.z);
	}

	CUDA_DEVICE_HOST void reset()
	{
		// default identity
		_row0 = make_float4(1,0,0,0);
		_row1 = make_float4(0,1,0,0);
		_row2 = make_float4(0,0,1,0);
	}

	CUDA_DEVICE_HOST SE3(float* arr)
	{
		_row0 = make_float4(arr[0], arr[1], arr[2], arr[3]);
		_row1 = make_float4(arr[4], arr[5], arr[6], arr[7]);
		_row2 = make_float4(arr[8], arr[9], arr[10], arr[11]);
	}

	CUDA_DEVICE_HOST SE3() {
		reset();
	}

//	friend ostream& operator<<(ostream& os, const SE3& se3);

};




class curvQE{

private:
	SE3 _E;
	float3 _C;

public:

	CUDA_DEVICE_HOST SE3 get_rot_mat() {
		return _E;
	}



	CUDA_DEVICE_HOST float3 get_rot_vector() {
		return _E.ln();
	}

	CUDA_DEVICE_HOST float get_z_trans() {
		return _E._row2.w;
	}


	CUDA_DEVICE_HOST float3 get_curvatures()
	{
		return _C;
	}

	const float max_curv = 0.1f;
	const float min_curv = -0.1f;

	CUDA_DEVICE_HOST float2 get_principal_curvatures()
	{
		float T1 = _C.x + _C.y;
		float T2 = _C.x*_C.y - _C.z*_C.z;

//		T1 /= 2;
		if((T1*T1 - 4*T2) < 1e-7)
			T2 = 0.0;
		else
			T2 = sqrt(T1*T1 - 4*T2);

		// keeps values sensible
		float2 curv = make_float2(T1+T2, T1-T2);
		curv.x = (curv.x > max_curv) ? max_curv : ((curv.x < min_curv) ? min_curv : curv.x);
		curv.y = (curv.y > max_curv) ? max_curv : ((curv.y < min_curv) ? min_curv : curv.y);

		//curv = make_float2(_C.z, _C.z);

		return curv;
	}

	CUDA_DEVICE_HOST float get_error(float4 & xp, const float weight=1.0f)
	{
		float4 xs = _E.mult(xp);
		return weight*(xs.x*xs.x*_C.x + 2*xs.x*xs.y*_C.z + xs.y*xs.y*_C.y + xs.z);
	}

	CUDA_DEVICE_HOST void init_frame(const float3 &normal)
	{
		float3 e_x = make_float3(1.0, 0.0, 0.0);

		float3 E_x_frame = e_x - normal*dot(e_x,normal);
		E_x_frame = normalize(E_x_frame);
		float3 E_y_frame = cross(normal, E_x_frame);
		E_y_frame = normalize(E_y_frame);

		_E._row0 = make_float4(E_x_frame.x, E_x_frame.y, E_x_frame.z, 0.0f);
		_E._row1 = make_float4(E_y_frame.x, E_y_frame.y, E_y_frame.z, 0.0f);
		_E._row2 = make_float4(normal.x, normal.y, normal.z, 0.0f);
	}


	// TODO - wrong, correct it
	CUDA_DEVICE_HOST void update_E_rot(WLS * wls)
	{
		float3 w = make_float3(wls->x0, wls->x1, 0);
		_E.update_rot(w);
	}

	CUDA_DEVICE_HOST float3 rotate(const float3 & vec)
	{
		return _E.rotate(vec);
	}

	CUDA_DEVICE_HOST void update(WLS *wls)
	{
		// rotation
		update_E_rot(wls);

		// z translation
		_E._row2.w += wls->x2;

		// curvature
		_C.x += wls->x3;
		_C.y += wls->x4;
		_C.z += wls->x5;
	}

	CUDA_DEVICE_HOST void compute_J(const float4 & xp,
			float* J)
	{
		float4 xs = _E.mult(xp);
		xs.z -= _E._row2.w;

		J[0] = -2*xs.z*(_C.y*xs.y+_C.z*xs.x)+xs.y;
		J[1] = 2*xs.z*(_C.x*xs.x+_C.z*xs.y)-xs.x;
		J[2] = 1.0f;
		J[3] = xs.x*xs.x;
		J[4] = xs.y*xs.y;
		J[5] = 2*xs.x*xs.y;

	}

	// copy constructor
	CUDA_DEVICE_HOST curvQE(const curvQE& QE)
	{
		this->_E._row0 = QE._E._row0;
		this->_E._row1 = QE._E._row1;
		this->_E._row2 = QE._E._row2;

		this->_C = QE._C;
	}

	CUDA_DEVICE_HOST void reset()
	{
		_E.reset();
		_C = make_float3(0,0,0);
	}

	CUDA_DEVICE_HOST curvQE(float* E_arr, float Cx, float Cy, float Cxy, float z_trans) :
			_E(E_arr),
			_C(make_float3(Cx, Cy, Cxy))
	{

	}

	CUDA_DEVICE_HOST curvQE() :
			_E(),
			_C(make_float3(0,0,0))
	{

	}

//	friend ostream& operator<<(ostream& os, const curvQE& QE);

};



class wls_LVM{

private:
	WLS _wls;
	curvQE _QE;
	SE3 _norm_update;

	int _max_iterations;
	float _lambda;

	// error info
	float3 _stats;
	float _total_err;
	int _num_points_used_in_patch;

	/// coordinate information
	float4 _center;
	int _center_idx;
	int _img_size;

public:

	__device__ float3 get_normal() {
		return make_float3(_QE.get_rot_mat()._row2.x, _QE.get_rot_mat()._row2.y, _QE.get_rot_mat()._row2.z);
	}

	__device__ float get_total_error(const float4* coords,
			curvQE* QE)
	{
		float total_error = 0.0f;
		float4 xp;
		for(int offset_idx = 0; offset_idx<num_offsets; ++offset_idx)
		{
			int shifted_idx = _center_idx + index_offsets[offset_idx];
			if(shifted_idx < 0 || shifted_idx >= _img_size || coords[shifted_idx].z == 0) continue;

			xp = coords[shifted_idx] - _center;
			xp.w = 1.0f; // remain homogeneous;

			float err = QE->get_error(xp);
			total_error += fabs(err);
		}

		return total_error;
	}


	__device__ float3 get_error_stats(const float4* coords,
			curvQE * QE)
	{
		float total_err = 0.0f, total_sqr_err = 0.0f, error, errsqr;
		int count = 0, shifted_idx;

		float4 xp;

		for(int offset_idx = 0; offset_idx<num_offsets; ++offset_idx)
		{
			shifted_idx = _center_idx + index_offsets[offset_idx];
			if(shifted_idx < 0 || shifted_idx >= _img_size || coords[shifted_idx].z == 0) continue;

			xp = coords[shifted_idx] - _center;
			xp.w = 1.0f;


			error = QE->get_error(xp);

//			if(abs(error) > 50) continue;
			if(isnan(error)) continue;

			total_err += error;

			errsqr = error*error;
			total_sqr_err += errsqr;
			count++;
		}

		float ave_sqr_err = total_sqr_err/(float)count;
		float ave_err = total_err/(float)count;
		float var = ave_sqr_err - ave_err*ave_err;

		return make_float3(ave_err, ave_sqr_err, var);
	}


	__device__ void compute_curvature(const float4* coords,
			float* J)
	{



		for(int i=0; i<_max_iterations; ++i)
		{
			_wls.init();
			_total_err = 0;
			_num_points_used_in_patch = 0;

//			_stats = get_error_stats(coords, &_QE);

			int points_used = 0;

			// iterate through patch
			for(int i = 0; i<num_offsets; ++i)
			{


				int shifted_idx = _center_idx + index_offsets[i];
				if(shifted_idx < 0 || shifted_idx >= _img_size || coords[shifted_idx].z == 0){
					continue;
				}

				float4 transformed_coord = coords[shifted_idx] - _center;

				float dist = mag(make_float3(transformed_coord.x, transformed_coord.y, transformed_coord.z));

				transformed_coord.w = 1.0f;


				// records shifted coords
				_QE.compute_J(transformed_coord, J);
				float error = _QE.get_error(transformed_coord);

				_total_err += error;

				float weight = 1.0f;

				weight = 16/(16 + dist*dist + error*error); /// Fixed here but can be variable

				if(error*error > 200.0) continue; /// Can be fixed or computed statistically
				//if(dist*dist > 200.0) continue; /// Can be used instead of error
//				if(error*error > _center.z) continue; /// works well in most cases

				_num_points_used_in_patch++;

				_wls.add_mJ_LVM(-error, J, weight, _lambda);


			}

			_wls.compute();
			_QE.update(&_wls);

			float3 w = make_float3(-_wls.x0,-_wls.x1, 0);
			if(w.x != 0.0 || w.y != 0.0)
				_norm_update.update_rot(w);
		}
	}

	CUDA_DEVICE_HOST void init_wls(){
		_wls.init();
	}

	CUDA_DEVICE_HOST void init_QE(const float3 & norm) {
		_QE.init_frame(norm);
	}

	CUDA_DEVICE_HOST WLS* get_WLS()
	{
		return &_wls;
	}

	CUDA_DEVICE_HOST curvQE* get_QE()
	{
		return &_QE;
	}

	CUDA_DEVICE_HOST SE3* get_norm_update()
	{
		return &_norm_update;
	}

	CUDA_DEVICE_HOST wls_LVM(const int &width, const int &height, const int center_idx,	const float4 &center_coord, int max_its=3, float lambda=0.0f) :
				_norm_update(),
				_total_err(0.0f),
				_num_points_used_in_patch(0),
				_lambda(lambda),
				_max_iterations(max_its)
	{
		_norm_update._row0 = make_float4(1,0,0,0);
		_norm_update._row1 = make_float4(0,1,0,0);
		_norm_update._row2 = make_float4(0,0,1,0);
		_center = center_coord;
		_center_idx = center_idx;
		_img_size = width*height;
	}

};

__host__ void update_index_offsets(int radius,
		const int width)
{

	using namespace std;

	int size = radius*2+1;


	int c_idx = radius+radius*size;

	cout << "Center: " << c_idx << endl;

	int * index_offsets_host = new int[size*size];

	cout << "PATCH" << endl;
	int index = 0;
	for(int i = -radius; i<=radius; ++i) {
		for(int j = -radius; j<=radius; ++j) {
			if((i*i+j*j) < (radius*radius+5)) {
				float x_diff = i;
				float y_diff = j;
				index_offsets_host[index++] = (i*width + j)*3;
				cout << "1 ";
			}
			else
				cout << "0 ";
		}
		cout << endl;
	}

//
	for(int i = 0; i<index; ++i)
		cout << index_offsets_host[i] << " ";
	cout << endl;

	/// udpate index offsets
	std::cout << "NUMBER OF OFFSETS: " << index << " size: " << sizeof(int)*index << std::endl;
	gl_cuda_vbo<float>::cudaCheck(cudaMemcpyToSymbol(index_offsets, index_offsets_host, sizeof(int)*index), "Copying Offsets");
	gl_cuda_vbo<float>::cudaCheck(cudaMemcpyToSymbol(num_offsets, &index, sizeof(int)), "Copying Num Offsets");
}

#endif /* SE3_H_ */
