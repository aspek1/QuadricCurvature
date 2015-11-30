/*
 * fileLoading.h
 * Used to read from files
 *  Created on: 29/10/2014
 *      Author: andrew
 */

#ifndef FILELOADING_H_
#define FILELOADING_H_

#include <vector>
#include <iostream>


void write_vect4_binary(const float4* values, std::string filename, int width, int height)
{
	using namespace std;
	ofstream output;
	output.open(filename.c_str(), ios::binary | ios::trunc);
	output.write((char*)values, sizeof(float4)*width*height);
	output.close();
}

void read_vect4_binary(std::string filename, std::vector<float4> &curv)
{
	using namespace std;

	curv.clear();

	ifstream input;
	input.open(filename.c_str(), ios::binary);
	float4 f;

	int vals = 0;
	while(input.read((char*)&f, sizeof(float4))){
		curv.push_back(f);
		vals++;
	}
	input.close();

	cout << "Loaded " << vals << " Values" << endl;

}

template<typename T>
void write_ascii(T values, std::string filename, int width, int height, int stride)
{
	using namespace std;

	int length = width*height;

	ofstream output;
	output.open(filename.c_str());
	for(int idx = 0; idx < length*stride; idx+=stride) {
		for(int s = 0; s<stride; ++s) {
			output << values[idx + s] << " ";
		}
		output << std::endl;
	}
	output.close();
}

void write_vect3_ascii(const float3* values, std::string filename, int width, int height)
{
	using namespace std;

	int length = width*height;

	ofstream output;
	output.open(filename.c_str());
	for(int idx = 0; idx < length; ++idx) {
		output << values[idx].x << " " << values[idx].y << " " << values[idx].z << std::endl;
	}
	output.close();
}

void write_vect3_binary(const float3* values, std::string filename, int width, int height)
{
	using namespace std;
	ofstream output;
	output.open(filename.c_str(), ios::binary | ios::trunc);
	output.write((char*)values, sizeof(float3)*width*height);
	output.close();
}

void read_vect3_binary(std::string filename, std::vector<float3> &curv)
{
	using namespace std;

	curv.clear();

	ifstream input;
	input.open(filename.c_str(), ios::binary);
	float3 f;

	int vals = 0;
	while(input.read((char*)&f, sizeof(float3))){
		curv.push_back(f);
		vals++;
	}
	input.close();

	cout << "Loaded " << vals << " Values" << endl;

}

#endif /* FILELOADING_H_ */
