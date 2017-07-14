# QuadricCurvature

Repository created for work performed for paper: A Fast Method For Computing Principal Curvatures From Range Images (ref below)

Open Source License: feel free to use all or part of this repo in any project in any way but please 
reference the following publication in any future work that uses it

Dataset(s) available from [Dropbox](https://www.dropbox.com/sh/c49rqfwnhgqcyv5/AAAZrxCpcuol4GzdSPgOtpa6a?dl=0)

Paper Links: 
[University Link](http://arrow.monash.edu.au/hdl/1959.1/1254327)
[arXiv](https://arxiv.org/pdf/1707.00381.pdf)

```
Cite: 
@inproceedings{Spek2012,
author = {Spek, Andrew and Drummond, Tom},
booktitle = {Australasian Conf. on Robitics and Automation (ACRA)},
title = {A Fast Method For Computing Principal Curvatures From Range Images},
year = {2015}
}
```

Please post any issues, and they will be fixed asap (if at all possible)

##Compiling/Building this Code

###REQUIRED:

CUDA Toolkit (latest) - https://developer.nvidia.com/cuda-toolkit

TooN (latest) - http://www.edwardrosten.com/cvd/toon.html

libCVD (latest) - http://www.edwardrosten.com/cvd/

###OPTIONAL:

OpenGL - https://www.opengl.org/

GLEW - http://glew.sourceforge.net/

*NOTE: if these are not included you will need to remove the GLKeyframe and gl_cuda_vbo classes in GLCudaInterop.hpp, and any function that uses GLKeyFrame objects.


###Windows (untested)

Simplest is to use Visual Studio and include all relevant libraries and include directories.

###Linux 

The simplest way by far is to import the project into the NSight eclipse IDE. The repository includes the relevant project files and will generate make files in order to build project. Alternatively build it manually, ensure to include c++11 flag  (-std=c++11) for build.

###OSX (untested)

 - TBA

##Using the Code

The provided example provides a simple base for computing curvature given input frames. We provide a file reading class that will read a sorted list of depth and color images from a dataset directory in the format provided in the sample dataset. It then generates a curvature image, and saves it to a file in row-major ascii format in the root directory. It will also optionally save the computed normals and coords (in millimetres) as determined by flags in the config.txt file.
