/*
 * configuration.h
 *
 *  Created on: 09/09/2014
 *      Author: andrew
 */

#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <map>

class configuration {

private:
	std::map<std::string, std::string> *config;

public:

	bool init_from_file(std::string filename = "config.txt", std::string dir = "./ConfigFiles/")
	{
		string line;
		std::string path = dir+filename;

		ifstream config_file (path.c_str());
		if (config_file.is_open())
		{
			while ( getline (config_file,line) ) {
				if(line[0] == '#') continue;

				int found = line.find_first_of("=");
				if(found <= 0 || found > line.size()-1) continue;

				// found config
				std::string key = line.substr(0, found);
				std::string value = line.substr(found+1, line.size()-1);

				std::cout << "Added Config : " << key << " -> " << value << std::endl;

				add_config_value(key, value);
			}
			config_file.close();
		}
		else{
			std::cerr << "Unable to open file";
			return false;
		}

	}

	bool containsKey(std::string key)
	{
		return !(*config)[key].empty();
	}

	std::string get_config_value(std::string key)
	{
		if(!containsKey(key)) {
			std::cerr << "Key not found in configuration" << std::endl;
			return "";
		}
		return (*config)[key];
	}

	bool get_bool(std::string key)
	{
		if(!containsKey(key)) {
			std::cerr << "Key not found in configuration" << key << std::endl;
			return false;
		}
		std::string s = (*config)[key];
		return s == "true";
	}

	int get_int(std::string key)
	{
		if(!containsKey(key)) {
			std::cerr << "Key not found in configuration" << key << std::endl;
			return 0.0f;
		}
		std::string s = (*config)[key];
		return (int)std::atoi(s.c_str());
	}

	float get_float(std::string key)
	{
		if(!containsKey(key)) {
			std::cerr << "Key not found in configuration" << key << std::endl;
			return 0.0f;
		}
		std::string s = (*config)[key];
		return (float)std::atof(s.c_str());
	}

	std::vector<float> get_float_vect(std::string key)
	{
		if(!containsKey(key)) {
			std::cerr << "Key not found in configuration" << key << std::endl;
			return std::vector<float>();
		}
		std::stringstream ss((*config)[key]);

		float f;
		std::vector<float> vals;

		while((ss >> f))
			vals.push_back(f);

		return vals;
	}

	bool add_config_value(std::string key, std::string value)
	{
		if(containsKey(key)){
			std::cerr << "Configuration already contains key" << key << std::endl;
			return false;
		}
		(*config)[key] = value;
		return true;
	}

	std::string operator[](std::string key) {
		return get_config_value(key);
	};

	configuration(){
		config = new std::map<std::string, std::string>();
	}
	~configuration(){

	}

};


#endif /* CONFIGURATION_H_ */
