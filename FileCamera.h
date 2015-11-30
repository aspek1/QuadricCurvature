/*
 * FileCamera.h
 *
 *  Created on: 18/09/2015
 *      Author: andrew
 */

#ifndef FILECAMERA_H_
#define FILECAMERA_H_

#include "Cam.h"
#include <string.h>
#include <cvd/image_io.h>
#include <stdio.h>
#include <dirent.h>

#include <iostream>

#include <algorithm>


using namespace CVD;
using namespace std;

class FileCamera : public Cam {

	string dir;
	string RgbDIR;
	string DepthDIR;

	std::vector<string> rgb_frames;
	std::vector<string> depth_frames;

	DIR *colorDir;
	DIR *depthDir;

public:

	bool SetReadDirectory(string dir);

	// camera method
	bool get_framedata(Rgb<byte>* &color_data, unsigned short* &depth_data);
	bool get_frame(Image<Rgb<byte> >& color_image, Image<unsigned short>& depth_image);

	FileCamera(string dir);
	virtual ~FileCamera();
};

#endif /* FILECAMERA_H_ */
