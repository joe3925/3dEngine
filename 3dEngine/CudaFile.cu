#include <cudafile.cuh>
#include <cudaIncludes.h>

__global__ void matrixVectorMultiplyKernel(float* matrix, float* vectors, float* results, int numVectors) {
	int index = threadIdx.x;
	float sum = 0.0f;

	for (int vec = 0; vec < numVectors; ++vec) {
		sum = 0.0f;
		for (int i = 0; i < 4; ++i) {
			sum += matrix[index * 4 + i] * vectors[vec * 4 + i];
		}
		results[vec * 4 + index] = sum;
	}
}
void projectTriangles3Dto2DWithCuda(const std::vector<triangle>& triangles, const float(&matrix)[16], std::vector<point2D>& outPts2D, float* d_matrix, float* d_vectors, float* d_results) {
	// Each triangle has 3 vertices, and each vertex has 4 components (x, y, z, w)
	int numTriangles = triangles.size();
	int numVertices = numTriangles * 3;
	std::vector<float> vertices(numVertices * 4); // Flatten array for all vertices
	std::vector<float> results(numVertices * 4, 0); // Flatten array for results

	// Populate vertices array from the triangles
	for (int t = 0; t < numTriangles; ++t) {
		for (int i = 0; i < 3; ++i) {
			const point* p = (i == 0) ? &triangles[t].p1 : (i == 1) ? &triangles[t].p2 : &triangles[t].p3;
			for (int j = 0; j < 4; ++j) {
				vertices[(t * 3 + i) * 4 + j] = p->Position[j];
			}
		}
	}

	// Copy data to the GPU
	cudaError_t error = cudaMemcpy(d_matrix, matrix, 16 * sizeof(float), cudaMemcpyHostToDevice);
	error = cudaMemcpy(d_vectors, vertices.data(), vertices.size() * sizeof(float), cudaMemcpyHostToDevice);

	matrixVectorMultiplyKernel << <numVertices / 3, 4 >> > (d_matrix, d_vectors, d_results, numVertices);

	cudaMemcpy(results.data(), d_results, results.size() * sizeof(float), cudaMemcpyDeviceToHost);

	outPts2D.clear();
	for (int i = 0; i < numVertices; ++i) {
		if (results[i * 4 + 3] != 0.0f) { // Perspective divide
			float x = results[i * 4 + 0] / results[i * 4 + 3];
			float y = results[i * 4 + 1] / results[i * 4 + 3];
			outPts2D.push_back(point2D(x, y));
		}
	}
}