#pragma once
# define M_PI           3.14159265358979323846

#include <gmtl/gmtl.h>
#include <windows.H>
#include <vector>
#include <string>
#include <iostream>
#include <wingdi.h>
#include <math.h>
#include <chrono>
#include <sstream>
#include <fstream>
#include <array>

#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>
#include <type_traits>
#include <algorithm> 
#include <thread>



#include <cuda_runtime.h>
#include "cudaIncludes.h"
#include "THpool.h"



mesh loadOBJ(const std::string& filename);
ThreadPool* createThreadPool(size_t size);

ThreadPool* createThreadPool(size_t size) {
	return new ThreadPool(size);
}

mesh CreateCube(float center_x, float center_y, float center_z, float edge_length) {
	float half_edge = edge_length / 2.0f;
	std::vector<point> vertices = {
		// Front face
		point(center_x - half_edge, center_y - half_edge, center_z + half_edge),
		point(center_x + half_edge, center_y - half_edge, center_z + half_edge),
		point(center_x + half_edge, center_y + half_edge, center_z + half_edge),
		point(center_x - half_edge, center_y + half_edge, center_z + half_edge),
		// Back face
		point(center_x - half_edge, center_y - half_edge, center_z - half_edge),
		point(center_x + half_edge, center_y - half_edge, center_z - half_edge),
		point(center_x + half_edge, center_y + half_edge, center_z - half_edge),
		point(center_x - half_edge, center_y + half_edge, center_z - half_edge),
	};

	std::vector<triangle> triArray;
	// Front face
	triArray.push_back(triangle(vertices[0], vertices[1], vertices[2]));
	triArray.push_back(triangle(vertices[2], vertices[3], vertices[0]));
	// Right face
	triArray.push_back(triangle(vertices[1], vertices[5], vertices[6]));
	triArray.push_back(triangle(vertices[6], vertices[2], vertices[1]));
	// Back face
	triArray.push_back(triangle(vertices[5], vertices[4], vertices[7]));
	triArray.push_back(triangle(vertices[7], vertices[6], vertices[5]));
	// Left face
	triArray.push_back(triangle(vertices[4], vertices[0], vertices[3]));
	triArray.push_back(triangle(vertices[3], vertices[7], vertices[4]));
	// Top face
	triArray.push_back(triangle(vertices[3], vertices[2], vertices[6]));
	triArray.push_back(triangle(vertices[6], vertices[7], vertices[3]));
	// Bottom face
	triArray.push_back(triangle(vertices[4], vertices[5], vertices[1]));
	triArray.push_back(triangle(vertices[1], vertices[0], vertices[4]));

	return mesh(triArray);
}



/*void DrawMesh(HDC hdc, mesh& Mesh, COLORREF color, double width, double height, camera& cam) {
	std::vector<std::array<POINT, 3>> localFixedPoints;
	// Calculate the number of full sets of 3 triangles
	int fullSets = Mesh.vertexList.size() / batchSize;

	for (int i = 0; i < fullSets; ++i) {
		std::vector<triangle> triangleBatch(Mesh.vertexList.begin() + i * batchSize, Mesh.vertexList.begin() + (i + 1) * batchSize);
		triangleWrapper(triangleBatch, width, height, cam, localFixedPoints,9);

		// Draw each triangle in the batch
		for (auto& points : localFixedPoints) {
			DrawTriangle(hdc, points, color);
		}
		localFixedPoints.clear();
	}

	int remainder = Mesh.vertexList.size() % batchSize;
	if (remainder > 0) {
		std::vector<triangle> triangleBatch(Mesh.vertexList.end() - remainder, Mesh.vertexList.end());
		triangleWrapper(triangleBatch, width, height, cam, localFixedPoints, 0);

		for (auto& points : localFixedPoints) {
			DrawTriangle(hdc, points, color);
		}
	}
}*/
mesh loadOBJ(const std::string& filename) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::vector <triangle> fail;
		return  mesh(fail);
	}

	std::vector<triangle> tri;
	std::vector<point> vertices;

	std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string prefix;
		iss >> prefix;
		if (prefix == "v") {
			// Vertex position
			float x, y, z;
			iss >> x >> y >> z;
			vertices.emplace_back(x, y, z);
		}
		else if (prefix == "f") {
			std::string vertex1, vertex2, vertex3;
			iss >> vertex1 >> vertex2 >> vertex3;

			int idx1 = std::stoi(vertex1.substr(0, vertex1.find('/')));
			int idx2 = std::stoi(vertex2.substr(0, vertex2.find('/')));
			int idx3 = std::stoi(vertex3.substr(0, vertex3.find('/')));

			// OBJ files are 1-indexed
			tri.emplace_back(vertices[idx1 - 1], vertices[idx2 - 1], vertices[idx3 - 1]);
		}

	}

	return mesh(tri);
}




