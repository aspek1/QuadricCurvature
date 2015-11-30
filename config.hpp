/*
 * config.hpp
 *
 *  Created on: 08/08/2014
 *      Author: andrew
 */

#ifndef CONFIG_HPP_
#define CONFIG_HPP_

// System Config
const int total_frames = 5;
int frame = 0;
int frames_capt = 0;
bool capt_frame = false;

// Running Config
bool filterDepth = true;
bool auto_cap = false;
bool capt_curvature = false;
bool smooth_normals = true;

// Offline running
bool inc_frame = false; 				// used to automatically step through test data

// Filter Config
int filter_r = 2;
float sigma_r = 5.0f;

float max_thresh = 4500.0f;
float min_thresh = 500.0f;

#endif /* CONFIG_HPP_ */
