#include <cudafile.cuh>
#include <cudaIncludes.h>

__global__ void matrixVectorMultiplyKernel(float* ProjMatrix, float* ViewMatrix, float* d_ViewResults, float* vectors, float* results, int numVectors, int* bytesReturned) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numVectors) {
		static int mbytesReturned;
		mbytesReturned = numVectors;
		// Temporary arrays to hold intermediate and final results for a single vector
		double tempResult[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

		// Multiply the vector by the view matrix and store the result in d_ViewResults
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				tempResult[i] += ViewMatrix[i * 4 + j] * vectors[idx * 4 + j];
			}
		}
		if (tempResult[2] < 0) {
			mbytesReturned--;
			return;
		}

		// Multiply the temporary result by the projection matrix and store directly in results
		for (int i = 0; i < 4; i++) {
			double finalValue = 0.0f;
			for (int j = 0; j < 4; j++) {
				finalValue += ProjMatrix[i * 4 + j] * tempResult[j];
			}
			results[idx * 4 + i] = finalValue;
		}
		__syncthreads();
		*bytesReturned = mbytesReturned;
		
	}
}

void copyProjMatrix(float* p_matrix, const float(&ProjMatrix)[16]) {
	memcpy(p_matrix, ProjMatrix, 16 * sizeof(float));
}

cudaError_t projectTriangles3Dto2DWithCuda(const std::vector<triangle>& triangles, const float(&ViewMatrix)[16], std::vector<point2D>& outPts2D, float* d_ViewResults, float* p_matrix, float* v_matrix, float* d_vectors, float* d_results, int* bytesReturned) {
	// Each triangle has 3 vertices, and each vertex has 4 components (x, y, z, w)
	int numTriangles = triangles.size();
	int numVertices = numTriangles * 3;
	int* bytesToRead = new(int);
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
	memcpy(d_vectors, vertices.data(), vertices.size() * sizeof(float));
	memcpy(v_matrix, ViewMatrix, 16 * sizeof(float));

	matrixVectorMultiplyKernel << <numVertices, 4 >> > (p_matrix, v_matrix, d_ViewResults, d_vectors, d_results, numVertices, bytesReturned);

	// Copy results back to host
	cudaMemcpy(bytesToRead, bytesReturned, sizeof(int), cudaMemcpyDeviceToHost);
	*bytesToRead *= 4;
	cudaMemcpy(results.data(), d_results, numVertices * sizeof(double) * 2, cudaMemcpyDeviceToHost);

	outPts2D.clear();
	for (int i = 0; i < results.size()/4; ++i) {
		if (results[i * 4 + 3] != 0.0f) { // Perspective divide
			float x = results[i * 4 + 0] / results[i * 4 + 3];
			float y = results[i * 4 + 1] / results[i * 4 + 3];
			outPts2D.push_back(point2D(x, y));
		}
		else {
			return cudaErrorUnknown; // Return an error if perspective divide fails
		}
	}
	return cudaSuccess;
}