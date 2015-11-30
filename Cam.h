/*
 * Cam.h
 *
 *  Created on: 18/09/2015
 *      Author: andrew
 */

#ifndef CAM_H_
#define CAM_H_

#include <cvd/image_io.h>

using namespace CVD;

class Cam {
public:

	int frame_idx;

	virtual bool get_framedata(Rgb<byte> * &color, unsigned short * &depth) = 0;
	virtual bool get_frame(Image<Rgb<byte> > &color, Image<unsigned short> &depth) = 0;
};

#endif /* CAM_H_ */
