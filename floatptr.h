/*
 * floatptr.h
 *
 *  Created on: 09/10/2015
 *      Author: andrew
 */

#ifndef FLOATPTR_H_
#define FLOATPTR_H_

#include <stdio.h>
#include <iostream>

class ref_count {

	int count;
public:
	inline int add_ref()
	{
		return count++;
	}

	inline int release()
	{
		return count--;
	}

};

class float_ptr {

private:
	float* ptr;
	int len;

	ref_count* ref;

public:

	float &operator[] (int idx);

	inline float* get_pointer() { return ptr; }
	inline int length() { return len; }

	float_ptr& operator = (const float_ptr& fp);


	float_ptr();
	float_ptr(int length);
	float_ptr(const float_ptr& pVal);

	virtual ~float_ptr();
};


#endif /* FLOATPTR_H_ */
