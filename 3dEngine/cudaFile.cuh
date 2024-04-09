#include "cudaIncludes.cuh"
extern "C" void projectTriangles3Dto2DWithCuda(const std::vector<triangle>&triangles, const float(&matrix)[16], std::vector<point2D>&outPts2D, float* d_matrix, float* d_vectors, float* d_results);
