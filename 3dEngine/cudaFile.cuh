#pragma once
#ifndef CUDAFILE_CU
#define CUDAFILE_CU

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudaIncludes.h>

class triangle;
class point2D;

extern "C" cudaError_t projectTriangles3Dto2DWithCuda(const std::vector<triangle>&triangles, const float(&ViewMatrix)[16], std::vector<point2D>&outPts2D, float* d_ViewResults, float* p_matrix, float* v_matrix, float* d_vectors, float* d_results, int* bytesReturned);
void copyProjMatrix(float* p_matrix, const float(&ProjMatrix)[16]);

#endif 