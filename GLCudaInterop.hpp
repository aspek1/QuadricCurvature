/*
 * GLCudaInterop.hpp
 *
 *  Created on: 21/08/2014
 *      Author: andrew
 */

#ifndef GLCUDAINTEROP_HPP_
#define GLCUDAINTEROP_HPP_



#include <GL/glew.h>
//#include <GL/glx.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

#include <string>
#include <vector>
#include <iostream>

#include <cstring>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda.h>

//#include "my_cuda_helpers.h"

typedef unsigned char uchar;
typedef unsigned short ushort;

class KeyFrame;
class GLKeyFrame;

extern void convert_depth_to_xyz(ushort* d_depth, float4* d_coords, int width, int height);

extern void compute_normals(KeyFrame * kf);

extern void compute_normals(GLKeyFrame * kf);

extern void perform_bilateral_filter(const ushort* d_input, ushort* d_output, int r, float sigmaR, int width, int height);


template<class TYPE>
class gl_cuda_vbo {
private:
	struct cudaGraphicsResource *_cudaBuffer;

	GLuint _vbo;
	int _width;
	int _height;
	int _depth;
	size_t _size;

	TYPE* _mapped_pointer;
	bool _mapped;

	std::string _name;



public:

	size_t get_size()
	{
		return _size;
	}

	static void cudaCheck(int errCode, const char* str = "")
	{
		if(errCode != CUDA_SUCCESS)
		{
			std::cerr << str <<  " >> CUDA ERROR: " << cudaGetErrorString((cudaError_t)errCode) << " - exiting" << std::endl;
			exit(errCode);
		}
//		std::cout << "gl_cuda_vbo Success" << std::endl;
	}

	TYPE* get_mapped_buffer()
	{
//		TYPE* mapped_pointer;

		if(!_mapped) {
			cudaCheck(cudaGraphicsMapResources(1, &_cudaBuffer, 0), "Mapping Resources");
			cudaCheck(cudaGraphicsResourceGetMappedPointer((void **)(&_mapped_pointer), &_size, _cudaBuffer), "Getting Mapped Resource Pointer");
			_mapped = true;
		}
		return _mapped_pointer;
	}

	bool unmap_buffer()
	{
		std::string msg = "Unmapping " + _name + " Buffer";
//		std::cout << "Mapped: " << _mapped << std::endl;
		if(_mapped)
			cudaCheck(cudaGraphicsUnmapResources(1, &_cudaBuffer, 0), msg.c_str());
		_mapped = false;
		return true;
	}

	GLuint& get_vbo()
	{
		unmap_buffer();
		return _vbo;
	}

	gl_cuda_vbo(int width, int height, int depth = 1, std::string name = "") : _width(width), _height(height), _depth(depth), _mapped(false), _name(name)
	{
		glGenBuffers(1, &_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, _vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(TYPE)*_width*_height, 0, GL_STREAM_DRAW);

		GLenum err = glGetError();
		if(err != GL_NO_ERROR) {
			std::cerr << "GL ERROR gl_cuda_vbo line " << __LINE__  << " - ERROR: " << gluErrorString(err) << std::endl;
			return;
		}
		cudaCheck(cudaGraphicsGLRegisterBuffer(&_cudaBuffer, _vbo, cudaGraphicsRegisterFlagsNone), "Registering Graphics Buffer - gl_cuda_vbo");

//		std::cout << "Created cuda_gl_vbo " << std::endl;
	}
	virtual ~gl_cuda_vbo()
	{

	}

};

//TODO - make memory keep track of host and device side dirty

template<class TYPE>
class cuda_array
{
private:

	TYPE* _d_pointer;
	TYPE* _h_pointer;

	bool _dirty_device;

	int _width;
	int _height;
	int _depth;

	void cudaCheck(int errCode, const char* str = "")
	{
		if(errCode != CUDA_SUCCESS)
		{
			std::cerr << str <<  " >> CUDA ERROR: " << cudaGetErrorString((cudaError_t)errCode) << " - exiting" << std::endl;
			exit(errCode);
		}
	}

public:

	enum ACCESS_TYPE { Host, Device, HostDevice };

	ACCESS_TYPE _access_type;

	TYPE* get_device_array()
	{

		if(_access_type == HostDevice) {
			_dirty_device = true;
			return _d_pointer;
		}
		if(_access_type == Device) return _d_pointer;
		std::cerr << "No Device Side Array" << std::endl;
		return NULL;

	}

	TYPE* get_host_array()
	{
		if(_access_type == HostDevice) {
			if(_dirty_device) cudaCheck(cudaMemcpy(_h_pointer, _d_pointer, _width*_height*_depth*sizeof(TYPE), cudaMemcpyDeviceToHost));
			_dirty_device = false;
			return _h_pointer;
		}
		if(_access_type == Host) {
			return _h_pointer;
		}
		std::cerr << "No Device Side Array" << std::endl;
		return NULL;
	}

	/// access elements
	TYPE &get(int x_idx, int y_idx, int z_idx)
	{
		if(_access_type == HostDevice){
			if(_dirty_device){
				cudaCheck(cudaMemcpy(_h_pointer, _d_pointer, _width*_height*_depth*sizeof(TYPE), cudaMemcpyDeviceToHost));
				_dirty_device = false;
			}

		}
		if(_access_type == Device) {
			std::cerr << "Not a Host Side Array" << std::endl;
		}
		return _h_pointer[x_idx + y_idx*_width + z_idx*_width*_height];

	}
	TYPE &get(int idx)
	{
		if(_access_type == HostDevice){
			if(_dirty_device){
				cudaCheck(cudaMemcpy(_h_pointer, _d_pointer, _width*_height*_depth*sizeof(TYPE), cudaMemcpyDeviceToHost));
				_dirty_device = false;
			}
		}
		if(_access_type == Device) {
			std::cerr << "Not a Host Side Array" << std::endl;
		}
		return _h_pointer[idx];

	}

	TYPE &operator[]( int idx) {

		if(_access_type == HostDevice){
			if(_dirty_device){
				cudaCheck(cudaMemcpy(_h_pointer, _d_pointer, _width*_height*_depth*sizeof(TYPE), cudaMemcpyDeviceToHost));
				_dirty_device = false;
			}
		}
		if(_access_type == Device) {
			std::cerr << "Not a Host Side Array" << std::endl;
		}
		return _h_pointer[idx];
	}

	void copy_data(TYPE* h_data, bool deep_copy = false)
	{
		_dirty_device = false;
		switch(_access_type) {
			case Host: {
				if(deep_copy) memcpy(_h_pointer, h_data, _width*_height*_depth*sizeof(TYPE));
				else _h_pointer = h_data;
			}break;
			case HostDevice: {
				if(deep_copy) memcpy(_h_pointer, h_data, _width*_height*_depth*sizeof(TYPE));
				else _h_pointer = h_data;
				cudaCheck(cudaMemcpy(_d_pointer, _h_pointer, _width*_height*_depth*sizeof(TYPE), cudaMemcpyHostToDevice));
			}break;
			case Device: {
				cudaCheck(cudaMemcpy(_d_pointer, h_data, _width*_height*_depth*sizeof(TYPE), cudaMemcpyHostToDevice));
			}break;
		}
	}

	/// Copy Constructor
	cuda_array(const cuda_array* &array)
	{

		this->_access_type = array->_access_type;

		this->_d_pointer = array->_d_pointer;
		this->_h_pointer = array->_h_pointer;

		this->_dirty_device = array->_dirty_device;

		this->_width = array->_width;
		this->_height = array->_height;
		this->_depth = array->_depth;
	}


	cuda_array(ACCESS_TYPE access_type, int width, int height, int depth = 1, bool fill = false, int initial_value = 0) :
		_width(width),
		_height(height),
		_depth(depth),
		_dirty_device(false),
		_access_type(access_type)
	{
		// instantiate _h_pointer
		if(_access_type == Host || _access_type == HostDevice) {
			_h_pointer = (TYPE*) malloc(width*height*depth*sizeof(TYPE));
			if(fill) memset(_h_pointer, initial_value, width*height*depth*sizeof(TYPE));
		}
		if(_access_type == HostDevice || _access_type == Device) {
			cudaCheck(cudaMalloc((void**) &_d_pointer, width*height*depth*sizeof(TYPE)));
			if(fill) cudaCheck(cudaMemset(_d_pointer, initial_value, width*height*depth*sizeof(TYPE)));
		}
	}

	virtual ~cuda_array()
	{
		std::cout << "ARRAY DESTROYED " << std::endl;
		// deconstruct
		switch (_access_type) {
			case Host:
				free(_h_pointer);
				break;
			case Device:
				cudaCheck(cudaFree(_d_pointer));
				break;
			default:
				free(_h_pointer);
				cudaCheck(cudaFree(_d_pointer));
				break;
		}

	}

};

class GLKeyFrame {

	typedef unsigned short ushort;
	typedef cuda_array<ushort> ushort_ca;

private:

	GLuint _vao;

	ushort_ca* _depth_array;
	ushort_ca* _filtered_depth_array;

	gl_cuda_vbo<uchar3>* _color;
	gl_cuda_vbo<float3>* _normals;
	gl_cuda_vbo<float4>* _coords;

	int _width;
	int _height;

	bool _filter_depth;
	bool _smooth_normals;

	void cudaCheck(int errCode, const char* str = "")
	{
		if(errCode != CUDA_SUCCESS)
		{
			std::cerr << str <<  " >> CUDA ERROR: " << cudaGetErrorString((cudaError_t)errCode) << " - exiting" << std::endl;
			exit(errCode);
		}
	}

public:

	int get_size()
	{
		return _width*_height;
	}

	//TODO - fix vao stuff

	inline void bind_vao()
	{
//		glBindVertexArray(_vao);
	}

	inline void unbind_vao() {
//		glBindVertexArray(0);
	}

	inline int get_height() { return this->_height; }
	inline int get_width() { return this->_width; }

	// Get Mapped Stuff
	uchar3* get_mapped_color() {
		bind_vao();
		return _color->get_mapped_buffer();
	}

	float3* get_mapped_normals() {
		bind_vao();
		return _normals->get_mapped_buffer();
	}

	float4* get_mapped_coords() {
		bind_vao();
		return _coords->get_mapped_buffer();
	}

	void map_all() {
		get_mapped_color();
		get_mapped_coords();
		get_mapped_normals();
	}

	/// Unmap when your done
	bool unmap_color() {
		bool success = _color->unmap_buffer();
		unbind_vao();
		return success;
	}

	bool unmap_normals() {
		bool success = _normals->unmap_buffer();
		unbind_vao();
		return success;
	}

	bool unmap_coords() {
		bool success = _coords->unmap_buffer();
		unbind_vao();
		return success;
	}

	bool unmap_all()
	{
		return unmap_color() && unmap_normals() && unmap_coords();
	}

	/// Use GL for display
	GLuint& get_GL_color() {
		bind_vao();
		return _color->get_vbo();
	}

	GLuint& get_GL_normals() {
		bind_vao();
		return _normals->get_vbo();
	}

	GLuint& get_GL_coords() {
		bind_vao();
		return _coords->get_vbo();
	}

	bool bind_GL_coords()
	{
		glBindBuffer(GL_ARRAY_BUFFER, _coords->get_vbo());
		GLenum err = glGetError();
		if(err == GL_NO_ERROR) return true;
		std::cerr << "GL ERROR bind_GL_coords line " << __LINE__  << " - ERROR: " << gluErrorString(err) << std::endl;
		return false;
	}

	bool bind_GL_color()
	{
		glBindBuffer(GL_ARRAY_BUFFER, _color->get_vbo());
		GLenum err = glGetError();
		if(err == GL_NO_ERROR) return true;
		std::cerr << "GL ERROR bind_GL_color line " << __LINE__  << " - ERROR: " << gluErrorString(err) << std::endl;
		return false;
	}
	bool bind_GL_normals()
	{
		glBindBuffer(GL_ARRAY_BUFFER, _normals->get_vbo());
		GLenum err = glGetError();
		if(err == GL_NO_ERROR) return true;
		std::cerr << "GL ERROR bind_GL_normals line " << __LINE__  << " - ERROR: " << gluErrorString(err) << std::endl;
		return false;
	}

	void initialise_from_point_cloud(float* coords_data, uchar* color_data)
	{
		this->copy_color_to_device(color_data);
		this->copy_data_to_coords(coords_data);
		compute_normals(this);
	}

	void initialise_from_depth(ushort* depth_data, uchar* color_data, float filter_r, float sigma_r)
	{

		this->copy_color_to_device(color_data);

		// filter depth?
		if(_filter_depth) {
			_depth_array->copy_data(depth_data, false);
			perform_bilateral_filter(_depth_array->get_device_array(), _filtered_depth_array->get_device_array(), filter_r, sigma_r, _width, _height);
			convert_depth_to_xyz(_filtered_depth_array->get_device_array(), this->get_mapped_coords(), this->_width, this->_height);
		}
		else
			this->copy_depth_to_coords(depth_data);

		// compute gradients here?

		// compute normals
		compute_normals(this);
	}

	void copy_data_to_coords(float* coord_data)
	{
		float4* mapped_coords = get_mapped_coords();
		cudaCheck(cudaMemcpy(mapped_coords, coord_data, this->_width*this->_height*sizeof(float4), cudaMemcpyHostToDevice));
	}

	void copy_depth_to_coords(ushort* depth_data)
	{
		float4* mapped_coords = get_mapped_coords();

		cudaMemcpy(_depth_array->get_device_array(), depth_data, this->_width*this->_height*sizeof(ushort), cudaMemcpyHostToDevice);

//		_depth_array->copy_data(depth_data, true);
		convert_depth_to_xyz(_depth_array->get_device_array(), mapped_coords, this->_width, this->_height);
	}

	void copy_color_to_device(uchar* color_data)
	{
		uchar3* mapped_color = get_mapped_color();
		cudaCheck(cudaMemcpy(mapped_color, color_data, this->_width*this->_height*sizeof(uchar3), cudaMemcpyHostToDevice));
	}

	bool gl_buffer_check()
	{
		if(!glIsBuffer(_coords->get_vbo()) || !glIsBuffer(_color->get_vbo()) || !glIsBuffer(_normals->get_vbo()) )
		{
			std::cerr << "Depth: " << (glIsBuffer(_coords->get_vbo()) ? "true" : "false") << std::endl;
			std::cerr << "Color: " << (glIsBuffer(_color->get_vbo()) ? "true" : "false") << std::endl;
			std::cerr << "Normal: " << (glIsBuffer(_normals->get_vbo()) ? "true" : "false") << std::endl;
			return false;
		}
		return true;
	}

	bool& smooth_normals()
	{
		return _smooth_normals;
	}

	GLKeyFrame(const int width, const int height) :
		_width(width),
		_height(height),
		_filter_depth(false),
		_smooth_normals(true)
	{
		glGenVertexArrays(1, &_vao);
		bind_vao();

		this->_color = new gl_cuda_vbo<uchar3>(width, height, 1, "color"); // possibly make into pyramid
		this->_normals = new gl_cuda_vbo<float3>(width, height, 1, "normals");
		this->_coords = new gl_cuda_vbo<float4>(width, height, 1, "depth");

		// TODO add color gradients and color image pyramid

		this->_depth_array = new ushort_ca(ushort_ca::HostDevice, width, height); // depth side array of
		this->_filtered_depth_array = new ushort_ca(ushort_ca::HostDevice, width, height); // used by bilateral filter

		unbind_vao();
	}

	/// Don't forget to define properly later
	virtual ~GLKeyFrame()
	{

		std::cout << "Destroying Keyframe" << std::endl;
		_depth_array->~cuda_array();
		_filtered_depth_array->~cuda_array();
		return;
	}
};

#include <memory>

class KeyFrame {

	typedef unsigned short ushort;
	typedef cuda_array<ushort> ushort_ca;

private:

	GLuint _vao;

	std::unique_ptr<ushort_ca > _depth_array;
	std::unique_ptr<ushort_ca > _filtered_depth_array;

	std::unique_ptr<cuda_array<uchar3> > _color;
	std::unique_ptr<cuda_array<float3> > _normals;
	std::unique_ptr<cuda_array<float4> > _coords;

	int _width;
	int _height;

	bool _filter_depth;
	bool _smooth_normals;

	void cudaCheck(int errCode, const char* str = "")
	{
		if(errCode != CUDA_SUCCESS)
		{
			std::cerr << str <<  " >> CUDA ERROR: " << cudaGetErrorString((cudaError_t)errCode) << " - exiting" << std::endl;
			exit(errCode);
		}
	}

public:

	int get_size()
	{
		return _width*_height;
	}

	//TODO - fix vao stuff


	inline int get_height() { return this->_height; }
	inline int get_width() { return this->_width; }

	// Get Mapped Stuff
	uchar3* get_mapped_color() {
		return _color.get()->get_device_array();
	}

	float3* get_mapped_normals() {
		return _normals.get()->get_device_array();
	}

	float4* get_mapped_coords() {
		return _coords.get()->get_device_array();
	}

	void copy_data_to_coords(float* coord_data)
	{
		cudaCheck(cudaMemcpy(get_mapped_coords(), coord_data, this->_width*this->_height*sizeof(float4), cudaMemcpyHostToDevice));
	}

	void copy_depth_to_coords(ushort* depth_data)
	{

		cudaMemcpy(_depth_array->get_device_array(), depth_data, this->_width*this->_height*sizeof(ushort), cudaMemcpyHostToDevice);

		//		_depth_array->copy_data(depth_data, true);
		convert_depth_to_xyz(_depth_array->get_device_array(), get_mapped_coords(), this->_width, this->_height);
	}

	void copy_color_to_device(uchar* color_data)
	{
		uchar3* mapped_color = get_mapped_color();
		cudaCheck(cudaMemcpy(get_mapped_color(), color_data, this->_width*this->_height*sizeof(uchar3), cudaMemcpyHostToDevice));
	}

	void initialise_from_point_cloud(float* coords_data, uchar* color_data)
	{
		this->copy_color_to_device(color_data);
		this->copy_data_to_coords(coords_data);
		compute_normals(this);
	}

	void initialise_from_depth(ushort* depth_data, uchar* color_data, float filter_r, float sigma_r)
	{

		this->copy_color_to_device(color_data);

		// filter depth?
		if(_filter_depth) {
			_depth_array->copy_data(depth_data, false);
			perform_bilateral_filter(_depth_array->get_device_array(), _filtered_depth_array->get_device_array(), filter_r, sigma_r, _width, _height);
			convert_depth_to_xyz(_filtered_depth_array->get_device_array(), get_mapped_coords(), this->_width, this->_height);
		}
		else
			this->copy_depth_to_coords(depth_data);

		// compute gradients here?

		// compute normals
		compute_normals(this);
	}



	bool& smooth_normals()
	{
		return _smooth_normals;
	}

	KeyFrame(const int width, const int height) :
		_width(width),
		_height(height),
		_filter_depth(false),
		_smooth_normals(true),
		_color(new cuda_array<uchar3> (cuda_array<uchar3>::Device, width, height)),
		_normals(new cuda_array<float3> (cuda_array<float3>::Device, width, height)),
		_coords(new cuda_array<float4> (cuda_array<float4>::Device, width, height)),
		_depth_array(new ushort_ca(ushort_ca::HostDevice, width, height)),
		_filtered_depth_array(new ushort_ca(ushort_ca::HostDevice, width, height))
	{

	}

	/// Don't forget to define properly later
	virtual ~KeyFrame()
	{

		std::cout << "Destroying Keyframe" << std::endl;
		_depth_array->~cuda_array();
		_filtered_depth_array->~cuda_array();
		return;
	}
};


#endif /* GLCUDAINTEROP_HPP_ */
