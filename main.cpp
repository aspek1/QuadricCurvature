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

#include "CurvSolver.h"

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

#include <memory>

void initGLEW()
{
	glewExperimental = GL_TRUE;

	GLenum err = glewInit();

	if(err != GLEW_OK) {
		std::cerr << "Error Initialising GLEW exiting" << std::endl;
	}
	int errCode = glGetError();
}

int main()
{
	using namespace CVD;
	using namespace std;


	configuration config;
	config.init_from_file();

	initGLEW();

	FileCamera* listener = new FileCamera(config["DATASET_DIR"]);

	int frame_width = config.get_int("FRAME_WIDTH");
	int frame_height = config.get_int("FRAME_HEIGHT");

	int radius = config.get_int("RADIUS");
	int max_its = config.get_int("MAX_ITS");

	float focal_x = config.get_float("focal_x");
	float focal_y = config.get_float("focal_y");
	float center_x = config.get_float("center_x");
	float center_y = config.get_float("center_y");
	float r_3 = config.get_float("r_3");
	float r_4 = config.get_float("r_4");


	CURVATURE::CurvSolver* solver = new CURVATURE::CurvSolver(frame_width, frame_height, radius, max_its);

	solver->update_cam_model(focal_x, focal_y, center_x, center_y, r_3, r_4);

	while(1) {
		CVD::Image<Rgb<CVD::byte> > color = CVD::Image<Rgb<CVD::byte> > (CVD::ImageRef(frame_width, frame_height));
		CVD::Image<unsigned short > depth = CVD::Image<unsigned short > (CVD::ImageRef(frame_width, frame_height));

		if(!listener->get_frame(color, depth)) {
			break;
		}

		solver->copy_rgbd((uint16_t*)depth.data(), (uint8_t*)color.data());

		std::unique_ptr<float> h_curvature(new float[frame_height*frame_width*3]);
		std::unique_ptr<float> h_normals(new float[frame_height*frame_width*3]);
		std::unique_ptr<float> h_coords(new float[frame_height*frame_width*4]);

		solver->compute();

		solver->get_curvature(h_curvature.get());
		solver->get_normals(h_normals.get());
		solver->get_coords(h_coords.get());



		if(config.get_bool("PRINT_NORMALS")) {

			write_ascii(h_normals.get(), get_filename(config["DATASET_DIR"], "normals", listener->frame_idx, "dat"), frame_width, frame_height, 3);
		}

		///// PRINT OUT CURVATURE

		if(config.get_bool("PRINT_CURVATURE")) {
			write_ascii(h_curvature.get(), get_filename(config["DATASET_DIR"], "curvature", listener->frame_idx, "dat"), frame_width, frame_height, 3);
		}

		///// PRINT OUT COORDINATES

		if(config.get_bool("PRINT_COORDS")) {
			write_ascii(h_coords.get(), get_filename(config["DATASET_DIR"], "coords", listener->frame_idx, "dat"), frame_width, frame_height, 4);
		}



		std::cout << "Finished Frame: " << listener->frame_idx << std::endl;

		listener->frame_idx++;

	}

	std::cout << "Processed " << listener->frame_idx-1  << " frames " << std::endl;

	return 0;
}

#endif
