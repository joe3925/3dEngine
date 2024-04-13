#pragma once
#ifndef CUDAFILES_CU
#define CUDAFILES_CU

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudaIncludes.h>

class triangle;
class point2D;

cudaError_t projectTriangles3Dto2DWithCuda(const std::vector<triangle>& triangles, const float(&ViewMatrix)[16], const float(&ProjMatrix)[16], std::vector<point2D>& outPts2D, float* d_ViewResults, float* v_matrix, float* p_matrix, float* d_vectors, float* d_results);

#endif 