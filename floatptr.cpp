/*
 * floatptr.cpp
 *
 *  Created on: 09/10/2015
 *      Author: andrew
 */

#include "floatptr.h"

float& float_ptr::operator [](const int idx) {
	if(idx>=this->len or idx<0) {
		std::cerr << "OUT OF BOUNDS ERROR: " << idx << ", MAX_LENGTH: " << len-1 << std::endl;
	}
	return this->ptr[idx];
}



float_ptr::float_ptr(): len(0), ref(0) {
	// TODO Auto-generated constructor stub
	ref = new ref_count();
	ref->add_ref();
}

float_ptr::float_ptr(const float_ptr& pVal): ptr(pVal.ptr), len(pVal.len), ref(pVal.ref) {
	// TODO Auto-generated constructor stub
	ref->add_ref();
}

float_ptr::float_ptr(int length) : len(length), ref(0) {
	// TODO Auto-generated constructor stub
	ptr = new float[length];
	ref = new ref_count();
	ref->add_ref();

}

float_ptr::~float_ptr() {
	// TODO Auto-generated destructor stub
	if(ref->release() == 0) {
		std::cout << "Deleted float_ptr" << std::endl;
		delete ptr;
		delete ref;
	}
}

float_ptr& float_ptr::operator = (const float_ptr& fp)
{
	if (this != &fp) // Avoid self assignment
	{
		if(ref->release() == 0)
		{
			delete ptr;
			delete ref;
		}

		ptr = fp.ptr;
		ref = fp.ref;
		ref->add_ref();
	}
	return *this;
}
