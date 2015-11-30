/// Simple program created for use as a development template 

//	abcdefghijklmnopqrtsuvwxyz
//	ABCDEFGHIJKLMNOPQRSTUVWXYZ

#ifndef MY_GL_MAIN_
#define MY_GL_MAIN_

#define TEST_PROG
#undef TEST_PRINT

#undef USE_KINECT
#undef TEST_IT
#undef TEST_SHARED

/// Local Includes

#include "Cam.h"
#include "FileCamera.h"

#include "cuda_ext_vecs.h"


#include "floatptr.h"
#include "fileHandling.h"

#include <iostream>
#include <stdio.h>

/// Image Handling
#include <cvd/image_io.h>

/// CUDA
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

/// Projective Geometry Maths
#include <TooN/TooN.h>
#include <TooN/se3.h>
#include <TooN/Cholesky.h>

#include <cvd/camera.h>

// constants
#include "config.hpp"
#include "configuration.h"

#include "my_cuda_helpers.h"

/// globals

/// external CUDA functions

extern void update_index_offsets(int radius, const int width);

extern void updateCameraModel(float fx, float fy, float cx, float cy, float r_3, float r_4);

extern void compute_iterative_curvature_host(float4* coords,
		float3* norms,
		float3* curvatures,
		const int width,
		const int height,
		const int kernel_radius,
		const int max_iteration,
		float4* errors);

extern void convert_depth_to_xyz(ushort* d_depth,
		float4* d_coords,
		int width,
		int height);
extern void compute_normals(float4* coords,
		float3* norms,
		const int width,
		const int height,
		bool smooth_normals=true);


std::string get_filename(std::string datasetDir, std::string prefix, int frame_num, std::string type)
{
	std::string path = datasetDir;
	char buffer[1024];
	int l = sprintf(buffer, "%s/%s%08d.%s", path.c_str(), prefix.c_str(), frame_num, type.c_str());
	return std::string(buffer, l);
}

int main()
{
	using namespace CVD;
	using namespace std;


	configuration config;
	config.init_from_file();

	FileCamera* listener = new FileCamera(config["DATASET_DIR"]);

	/// TIMING TEST

	int frame_width = config.get_int("FRAME_WIDTH");
	int frame_height = config.get_int("FRAME_HEIGHT");

	float focal_x = config.get_float("focal_x");
	float focal_y = config.get_float("focal_y");
	float center_x = config.get_float("center_x");
	float center_y = config.get_float("center_y");
	float r_3 = config.get_float("r_3");
	float r_4 = config.get_float("r_4");

	int radius = config.get_int("RADIUS");


	/// Init Device Arrays
	unsigned short* d_depth = (unsigned short*)cudaArray<unsigned short>(frame_width*frame_height, sizeof(unsigned short), true);
	float4* d_error = (float4*)cudaArray<float4>(frame_width*frame_height, sizeof(float4), true);
	float3* d_curv = (float3*)cudaArray<float3>(frame_width*frame_height, sizeof(float3), true);
	float4* d_coords = (float4*)cudaArray<float4>(frame_width*frame_height, sizeof(float4), true);
	float3* d_norms = (float3*)cudaArray<float3>(frame_width*frame_height, sizeof(float3), true);
	uchar3* d_color = (uchar3*)cudaArray<uchar3>(frame_width*frame_height, sizeof(uchar3), true);

	unsigned short* d_depth_filtered = (unsigned short*)cudaArray<unsigned short>(frame_width*frame_height, sizeof(unsigned short), true);

	float3* d_color_gradients = (float3*)cudaArray<float3>(frame_width*frame_height, sizeof(float3), true);
	float3* d_depth_gradients = (float3*)cudaArray<float3>(frame_width*frame_height, sizeof(float3), true);

	float* d_border = (float*)cudaArray<float>(frame_width*frame_height, sizeof(float), true);

	/// Create Image
	CVD::Image<Rgb<CVD::byte> > color = CVD::Image<Rgb<CVD::byte> > (CVD::ImageRef(frame_width, frame_height));
	CVD::Image<unsigned short > depth = CVD::Image<unsigned short > (CVD::ImageRef(frame_width, frame_height));

	std::cout << "Computing Curvature From Directory: " << std::endl;
	std::cout << config["DATASET_DIR"] << std::endl << std::endl;


	updateCameraModel(focal_x, focal_y, center_x, center_y,  r_3, r_4);
	update_index_offsets(radius, frame_width);

	/// Start on frame 0
	listener->frame_idx=0;


	float4* h_coords2 = new float4[frame_width*frame_height];

	float_ptr h_curv(frame_width*frame_height*3);
	float_ptr h_norms(frame_width*frame_height*3);
	float_ptr h_quad(frame_width*frame_height*13);
	float_ptr h_coords(frame_width*frame_height*4);

	while(1) {

		int max_its = config.get_int("MAX_ITS");

		if(!listener->get_frame(color, depth)) {
			break;
		}

		/// Convert Input Depth to Point-cloud
		cudaMemcpy(d_depth, depth.data(), frame_width*frame_height*sizeof(unsigned short), cudaMemcpyHostToDevice);
		convert_depth_to_xyz(d_depth, d_coords, frame_width, frame_height);

		/// compute normals
		compute_normals(d_coords, d_norms, frame_width, frame_height, false);
		cudaMemset(d_curv, 0, sizeof(float3)*frame_width*frame_height);

		/// Compute Curvature
		compute_iterative_curvature_host(d_coords,
				d_norms,
				d_curv,
				frame_width,
				frame_height,
				radius,
				max_its,
				d_error);


		///// PRINT OUT NORMALS

		if(config.get_bool("PRINT_NORMALS")) {

			cudaCheck(cudaMemcpy(h_norms.get_pointer(), d_norms, sizeof(float3)*frame_width*frame_height, cudaMemcpyDeviceToHost));

			write_ascii(h_norms, get_filename(config["DATASET_DIR"], "normals", listener->frame_idx, "dat"), frame_width, frame_height, 3);
		}

		///// PRINT OUT CURVATURE

		if(config.get_bool("PRINT_CURVATURE")) {

			cudaMemcpy(h_curv.get_pointer(), d_curv, frame_width*frame_height*sizeof(float3), cudaMemcpyDeviceToHost);

			write_ascii(h_curv, get_filename(config["DATASET_DIR"], "curvature", listener->frame_idx, "dat"), frame_width, frame_height, 3);
		}

		///// PRINT OUT COORDINATES

		if(config.get_bool("PRINT_CURVATURE")) {
			cudaMemcpy(h_coords.get_pointer(), d_coords, frame_width*frame_height*sizeof(float4), cudaMemcpyDeviceToHost);

			write_ascii(h_coords, get_filename(config["DATASET_DIR"], "coords", listener->frame_idx, "dat"), frame_width, frame_height, 4);
		}



		std::cout << "Finished Frame: " << listener->frame_idx << std::endl;

		listener->frame_idx++;
	}
	

	std::cout << "Processed " << listener->frame_idx-1  << " frames " << std::endl;

	return 0;
}

#endif
