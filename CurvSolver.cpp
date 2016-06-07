/*
 * CurvSolver.cpp
 *
 *  Created on: 25 Apr 2016
 *      Author: andrew
 */

#include "CurvSolver.h"

namespace CURVATURE {

	void CurvSolver::copy_rgbd(const uint16_t* depth_array, const uint8_t* color) {

		/// filter and sigma constant
		_frame->initialise_from_depth((ushort*)depth_array, (uchar*)color, 0.2, 0.3);
	}

	void CurvSolver::compute(){

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);

		compute_iterative_curvature_host(_frame->get_mapped_coords(), _frame->get_mapped_normals(), (float3*)_curv, Width, Height, _radius, _max_its, (float4*)_error);
		cudaDeviceSynchronize();

		cudaEventRecord(stop);

		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
	#ifdef DEBUG
		std::cout << "Took: " << milliseconds << "ms "<< std::endl;
	#endif

	}

	void CurvSolver::get_curvature(float* curvature){
		cudaMemcpy(curvature, _curv, sizeof(float3)*Width*Height, cudaMemcpyDeviceToHost);
	}

	void CurvSolver::get_normals(float* normals) {
		cudaMemcpy(normals, _frame->get_mapped_normals(), sizeof(float3)*Width*Height, cudaMemcpyDeviceToHost);
	}

	void CurvSolver::get_coords(float* coords) {
		cudaMemcpy(coords, _frame->get_mapped_coords(), sizeof(float3)*Width*Height, cudaMemcpyDeviceToHost);
	}

	CurvSolver::CurvSolver(int width, int height, int radius, int max_its) :
				_radius(radius),
				_max_its(max_its),
				Width(width),
				Height(height)
	{
		// TODO Auto-generated constructor stub
		_frame = new KeyFrame(width, height);

		/// cuda memory
		cudaMalloc(&_curv, sizeof(float3)*Width*Height);
		cudaMalloc(&_error, sizeof(float4)*Width*Height);

		/// must be done for correct operation
		update_sample_patch();
	}

	CurvSolver::~CurvSolver() {
		// TODO Auto-generated destructor stub
	}

}
