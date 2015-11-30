/*
 * FileCamera.cpp
 *
 *  Created on: 18/09/2015
 *      Author: andrew
 */

#include <stddef.h>	/* size_t */
#include <ctype.h>

#include "FileCamera.h"


///////////////////////////////////////////////////////////////////////////////////////////

/*
 * nat_sort comparison adapted to C++ from http://sourcefrog.net/projects/
 */


/* These are defined as macros to make it easier to adapt this code to
 * different characters types or comparison functions. */

static inline int
nat_isdigit(char a)
{
     return isdigit((unsigned char) a);
}


static inline int
nat_isspace(char a)
{
     return isspace((unsigned char) a);
}


static inline char
nat_toupper(char a)
{
     return toupper((unsigned char) a);
}


static int
compare_right(char const *a, char const *b)
{
     int bias = 0;

     /* The longest run of digits wins.  That aside, the greatest
	value wins, but we can't know that it will until we've scanned
	both numbers to know that they have the same magnitude, so we
	remember it in BIAS. */
     for (;; a++, b++) {
	  if (!nat_isdigit(*a)  &&  !nat_isdigit(*b))
	       return bias;
	  if (!nat_isdigit(*a))
	       return -1;
	  if (!nat_isdigit(*b))
	       return +1;
	  if (*a < *b) {
	       if (!bias)
		    bias = -1;
	  } else if (*a > *b) {
	       if (!bias)
		    bias = +1;
	  } else if (!*a  &&  !*b)
	       return bias;
     }

     return 0;
}


static int
compare_left(char const *a, char const *b)
{
     /* Compare two left-aligned numbers: the first to have a
        different value wins. */
     for (;; a++, b++) {
	  if (!nat_isdigit(*a)  &&  !nat_isdigit(*b))
	       return 0;
	  if (!nat_isdigit(*a))
	       return -1;
	  if (!nat_isdigit(*b))
	       return +1;
	  if (*a < *b)
	       return -1;
	  if (*a > *b)
	       return +1;
     }

     return 0;
}


static int
strnatcmp0(char const *a, char const *b, int fold_case)
{
     int ai, bi;
     char ca, cb;
     int fractional, result;

     ai = bi = 0;
     while (1) {
	  ca = a[ai]; cb = b[bi];

	  /* skip over leading spaces or zeros */
	  while (nat_isspace(ca))
	       ca = a[++ai];

	  while (nat_isspace(cb))
	       cb = b[++bi];

	  /* process run of digits */
	  if (nat_isdigit(ca)  &&  nat_isdigit(cb)) {
	       fractional = (ca == '0' || cb == '0');

	       if (fractional) {
		    if ((result = compare_left(a+ai, b+bi)) != 0)
			 return result;
	       } else {
		    if ((result = compare_right(a+ai, b+bi)) != 0)
			 return result;
	       }
	  }

	  if (!ca && !cb) {
	       /* The strings compare the same.  Perhaps the caller
                  will want to call strcmp to break the tie. */
	       return 0;
	  }

	  if (fold_case) {
	       ca = nat_toupper(ca);
	       cb = nat_toupper(cb);
	  }

	  if (ca < cb)
	       return -1;

	  if (ca > cb)
	       return +1;

	  ++ai; ++bi;
     }
}


inline int
strnatcmp(char const *a, char const *b) {
     return strnatcmp0(a, b, 0);
}


/* Compare, recognizing numeric string and ignoring case. */
inline int
strnatcasecmp(char const *a, char const *b) {
     return strnatcmp0(a, b, 1);
}

inline bool strnatcmp_s(const string &a, const string &b) {
     return strnatcmp0(a.c_str(), b.c_str(), 0)<0;
}


/* Compare, recognizing numeric string and ignoring case. */
inline bool strnatcasecmp_s(const string &a, const string &b) {
     return strnatcmp0(a.c_str(), b.c_str(), 1)<0;
}


///////////////////////////////////////////////////////////////////////////////////////////

bool FileCamera::SetReadDirectory(string dir)
{
	bool color_found = true;
	bool depth_found = true;

	rgb_frames.clear();
	depth_frames.clear();

	this->dir = dir;

	this->RgbDIR = dir + "/rgb";
	this->colorDir = opendir (RgbDIR.c_str());


	if (colorDir == NULL) {
		cout << "Cannot open directory: " << dir << endl;
		color_found = false;
	}
	else {
		struct dirent *colorDirent;
		while((colorDirent = readdir(colorDir))!=NULL) {
			if(strcmp(colorDirent->d_name, ".") && strcmp(colorDirent->d_name, "..")) {
				rgb_frames.push_back(string(colorDirent->d_name));
				cout << "COLOR FILE: " << colorDirent->d_name << endl;
			}
		}
		sort(rgb_frames.begin(), rgb_frames.end(), strnatcmp_s);
	}

	closedir(this->colorDir);

	this->DepthDIR = dir + "/depth";
	this->depthDir = opendir (DepthDIR.c_str());

	if (depthDir == NULL) {
		cout << "Cannot open directory: " << dir << endl;
		depth_found = false;
	}
	else {
		struct dirent *depthDirent;
		while((depthDirent = readdir(depthDir))!=NULL) {
			if(strcmp(depthDirent->d_name, ".") && strcmp(depthDirent->d_name, "..")) {
				depth_frames.push_back(string(depthDirent->d_name));
				cout << "DEPTH FILE: " << depthDirent->d_name << endl;
			}
		}
		sort(depth_frames.begin(), depth_frames.end(), strnatcmp_s);
	}

	closedir(this->depthDir);

	frame_idx = 0;

	cout << "FILE CAMERA" << endl;
	cout << "RGB IMAGES: " << rgb_frames.size() << endl;
	cout << "DEPTH IMAGES: " << depth_frames.size() << endl;

	return depth_found;
}

bool FileCamera::get_framedata(Rgb<byte>* &color_data, unsigned short* &depth_data)
{
	try {
		if(frame_idx < rgb_frames.size()){
			cout << "Loading DATA" << endl;
			string rgb_filename = this->RgbDIR + "/" + rgb_frames[frame_idx];
			Image<Rgb<byte> > color = img_load(rgb_filename);
			memcpy(color_data, color.data(), color.size().x*color.size().y*sizeof(Rgb<byte>));
			cout << "Loaded Color Data" << endl;
		}
	}
	catch(CVD::Exceptions::Image_IO::OpenError err) {
		cerr << "Error in loaded color: " << err.what << endl;
	}
	catch(CVD::Exceptions::Image_IO::EofBeforeImage err) {
			cerr << "Error in loading depth: " << err.what << endl;
		}
	try {
		if(frame_idx < depth_frames.size()) {
			string depth_filename = this->DepthDIR + "/" + depth_frames[frame_idx];
			Image<unsigned short> depth = img_load(depth_filename);
			memcpy(depth_data, depth.data(), depth.size().x*depth.size().y*sizeof(unsigned short));
			cout << "Loaded Depth Data" << endl;
		}
		else return false;
	}
	catch(CVD::Exceptions::Image_IO::OpenError err) {
		cerr << "Error in loading depth: " << err.what << endl;
	}
	catch(CVD::Exceptions::Image_IO::EofBeforeImage err) {
		cerr << "Error in loading depth: " << err.what << endl;
	}
//	frame_idx++;

	return true;
}

bool FileCamera::get_frame(Image<Rgb<byte> >& color_image, Image<unsigned short>& depth_image)
{
	try {
		if(frame_idx < rgb_frames.size()){
			string rgb_filename = this->RgbDIR + "/" + rgb_frames[frame_idx];
			img_load(color_image, rgb_filename);
		}
	}
	catch(CVD::Exceptions::Image_IO::OpenError err) {
		cerr << "Error in loaded color: " << err.what << endl;
	}
	catch(CVD::Exceptions::Image_IO::EofBeforeImage err) {
			cerr << "Error in loading depth: " << err.what << endl;
	}
	if(frame_idx < depth_frames.size()) {
		string depth_filename = this->DepthDIR + "/" + depth_frames[frame_idx];

		try {
			img_load(depth_image, depth_filename);
		}
		catch(CVD::Exceptions::Image_IO::OpenError err) {
			cerr << "Error in loading depth: " << err.what << " " << depth_filename << endl;
		}
		catch(CVD::Exceptions::Image_IO::EofBeforeImage err) {
			cerr << "Error in loading depth: " << err.what << " " << depth_filename << endl;
		}
	}
	else return false;

//	frame_idx++;

	return true;
}

FileCamera::FileCamera(string dir) {
	// TODO Auto-generated 	constructor stub
	this->frame_idx = 0;
	this->SetReadDirectory(dir);
}

FileCamera::~FileCamera() {
	// TODO Auto-generated destructor stub
	rgb_frames.~vector();
	depth_frames.~vector();


}

