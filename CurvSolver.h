/*
 * CurvSolver.h
 *
 *  Created on: 25 Apr 2016
 *      Author: andrew
 *
 *  Class created to make object that computes curvature/normals on demand from depth images
 */

#ifndef CURVSOLVER_H_
#define CURVSOLVER_H_

#define DEBUG

#include "GLCudaInterop.hpp"
#include <cuda_runtime.h>
#include <chrono>

extern void compute_iterative_curvature_host(float4* coords, float3* norms, float3* curvatures,	const int width, const int height, const int kernel_radius,	const int max_iteration, float4* errors);
extern void updateCameraModel(float fx, float fy, float cx, float cy, float r_3, float r_4);
extern __host__ void update_index_offsets(int radius, const int width);


class timer {
private:
	std::chrono::high_resolution_clock::time_point _start;

public:

	inline void start() {
		_start = std::chrono::high_resolution_clock::now();
	}

	inline void check() {
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end-_start).count())/1e6 << "ms" << std::endl;
	}

	timer() {
		_start = std::chrono::high_resolution_clock::now();
	}

};

namespace CURVATURE {

	class CurvSolver {

		KeyFrame *_frame;
		int _radius;
		int _max_its;

		/// device side
		float* _curv;
		float* _error;

		inline void update_sample_patch(){
			update_index_offsets(_radius, Width);
		}

	public:

		int Width;
		int Height;

		void copy_rgbd(const uint16_t* depth_array, const uint8_t* color);

		/// preform compute
		void compute();

		/// for visualization or other purposes
		void get_curvature(float* curvature);
		void get_normals(float* normals);
		void get_coords(float* coords);

		inline void update_cam_model(float fx, float fy, float cx, float cy, float r2, float r4) {
			/// update __constant__ memory objects
			updateCameraModel(fx, fy, cx, cy, r2, r4);
		}


		CurvSolver(int height, int width, int radius=10, int max_its=10);
		virtual ~CurvSolver();
	};
}

#endif /* CURVSOLVER_H_ */
