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



#include "THpool.cuh"
#include "cudaFile.cuh"
#include <cuda_runtime.h>
#include "cudaIncludes.cuh"



void DrawMesh(HDC hdc, mesh& Mesh, COLORREF color, double width, double height, camera& cam, ThreadPool* pool);
mesh loadOBJ(const std::string& filename);
void rotate(mesh& Mesh, float x, float y, float z);
void transform(mesh& Mesh, float x, float y, float z);
gmtl::Vec4f Center(mesh& Mesh);
void moveCam(camera& cam, float moveSpeed);
POINT ConvertFromPoint2D(point2D& pt2D);
void triangleWrapper(mesh& Mesh, std::vector<triangle>& triangles, float width, float height, camera& cam, std::vector<std::array<POINT, 3>>& fixed, int currentThread);
void DrawTriangle(HDC hdc, const std::array<POINT, 3>& pArray, COLORREF color);
void fixPoint(point2D& p, int width, int height);
gmtl::Vec4f translateRotateTranslate(const gmtl::Vec4f& position, const gmtl::Vec4f& center, const gmtl::Matrix44f& rotationMatrix);
void rotateCam(camera& cam, float rotateSpeed);

ThreadPool* createThreadPool(size_t size);

void triangleWrapper(mesh &Mesh, std::vector<triangle>& triangles, float width, float height, camera& cam, std::vector<std::array<POINT, 3>>& fixed, int currentThread) {
	// Calculate the total number of vertices
	int totalVertices = triangles.size() * 3;
	std::vector<point2D> arg3D({{0,0},{0,0},{0,0},{0,0},{0,0},{0,0} ,{0,0},{0,0},{0,0} }); // Adjusted for multiple triangles

	// Process the triangles' vertices through CUDA
	projectTriangles3Dto2DWithCuda(triangles, cam.projectionMatrix.mData, arg3D, Mesh.mData.M_data[currentThread], Mesh.mData.V_data[currentThread], Mesh.mData.R_data[currentThread]);

	// Iterate over all triangles
	for (size_t t = 0; t < triangles.size(); ++t) {
		// Fix the points for each triangle (-1 - 1 to 0 - width, 0 - height)
		std::array<POINT, 3> points;
		for (int i = 0; i < 3; ++i) {
			fixPoint(arg3D[t * 3 + i], width, height); // Adjust index for flat array
			points[i] = ConvertFromPoint2D(arg3D[t * 3 + i]);
		}

		fixed.push_back(points);
	}
}

void setBatchSize(size_t size, mesh &Mesh) {
	Mesh.batchSize = size;
}

ThreadPool* createThreadPool(size_t size) {
	return new ThreadPool(size);
}

void DrawTriangle(HDC hdc, const std::array<POINT, 3>& pArray, COLORREF color) {
	HPEN hPen = CreatePen(PS_SOLID, 1, color);
	HPEN hOldPen = (HPEN)SelectObject(hdc, hPen);
	// Draw the triangle
	Polyline(hdc, pArray.data(), 3);

	SelectObject(hdc, hOldPen);
	DeleteObject(hPen);
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

	mesh cube_mesh;
	// Front face
	cube_mesh.vertexList.push_back(triangle(vertices[0], vertices[1], vertices[2]));
	cube_mesh.vertexList.push_back(triangle(vertices[2], vertices[3], vertices[0]));
	// Right face
	cube_mesh.vertexList.push_back(triangle(vertices[1], vertices[5], vertices[6]));
	cube_mesh.vertexList.push_back(triangle(vertices[6], vertices[2], vertices[1]));
	// Back face
	cube_mesh.vertexList.push_back(triangle(vertices[5], vertices[4], vertices[7]));
	cube_mesh.vertexList.push_back(triangle(vertices[7], vertices[6], vertices[5]));
	// Left face
	cube_mesh.vertexList.push_back(triangle(vertices[4], vertices[0], vertices[3]));
	cube_mesh.vertexList.push_back(triangle(vertices[3], vertices[7], vertices[4]));
	// Top face
	cube_mesh.vertexList.push_back(triangle(vertices[3], vertices[2], vertices[6]));
	cube_mesh.vertexList.push_back(triangle(vertices[6], vertices[7], vertices[3]));
	// Bottom face
	cube_mesh.vertexList.push_back(triangle(vertices[4], vertices[5], vertices[1]));
	cube_mesh.vertexList.push_back(triangle(vertices[1], vertices[0], vertices[4]));

	return cube_mesh;
}


void transform(mesh& Mesh, float x, float y, float z) {
	for (int i = 0; i < Mesh.vertexList.size(); i++) {

		Mesh.vertexList[i].p1.Position[0] += x;
		Mesh.vertexList[i].p1.Position[1] += y;
		Mesh.vertexList[i].p1.Position[2] += z;

		Mesh.vertexList[i].p2.Position[0] += x;
		Mesh.vertexList[i].p2.Position[1] += y;
		Mesh.vertexList[i].p2.Position[2] += z;

		Mesh.vertexList[i].p3.Position[0] += x;
		Mesh.vertexList[i].p3.Position[1] += y;
		Mesh.vertexList[i].p3.Position[2] += z;
	}
}
gmtl::Vec4f Center(mesh& Mesh) {
	gmtl::Vec4f center(0.0f, 0.0f, 0.0f, 1.0f);
	for (const triangle& tri : Mesh.vertexList) {
		center += tri.p1.Position + tri.p2.Position + tri.p3.Position;
	} // Average center position
	return center /= (Mesh.vertexList.size() * 3);
}
void rotate(mesh& Mesh, float x, float y, float z) {
	// Calculate the center of the mesh
	gmtl::Vec4f center = Center(Mesh);

	// Create rotation matrices for X, Y, and Z axes
	gmtl::Matrix44f rotationMatrixX, rotationMatrixY, rotationMatrixZ;
	gmtl::setRot(rotationMatrixX, gmtl::AxisAngle<float>(gmtl::Math::deg2Rad(x), 1.0f, 0.0f, 0.0f)); // Rotate around X axis
	gmtl::setRot(rotationMatrixY, gmtl::AxisAngle<float>(gmtl::Math::deg2Rad(y), 0.0f, 1.0f, 0.0f)); // Rotate around Y axis
	gmtl::setRot(rotationMatrixZ, gmtl::AxisAngle<float>(gmtl::Math::deg2Rad(z), 0.0f, 0.0f, 1.0f)); // Rotate around Z axis

	// Combine the rotations (order matters)
	gmtl::Matrix44f combinedRotationMatrix = rotationMatrixZ * rotationMatrixY * rotationMatrixX;

	// move the mesh to the orgin rotate it and move it back
	for (triangle& tri : Mesh.vertexList) {
		tri.p1.Position = translateRotateTranslate(tri.p1.Position, center, combinedRotationMatrix);
		tri.p2.Position = translateRotateTranslate(tri.p2.Position, center, combinedRotationMatrix);
		tri.p3.Position = translateRotateTranslate(tri.p3.Position, center, combinedRotationMatrix);
	}
}

gmtl::Vec4f translateRotateTranslate(const gmtl::Vec4f& position, const gmtl::Vec4f& center, const gmtl::Matrix44f& rotationMatrix) {
	// Translate to origin
	gmtl::Vec4f translatedPosition = position - center;
	// Rotate
	translatedPosition = rotationMatrix * translatedPosition;
	// Translate back
	translatedPosition += center;
	return translatedPosition;
}




POINT ConvertFromPoint2D(point2D& pt2D) {
	POINT pt;
	pt.x = static_cast<LONG>(pt2D.x);
	pt.y = static_cast<LONG>(pt2D.y);
	return pt;
}

void fixPoint(point2D& p, int width, int height) {
	p.x = (p.x + 1.0f) * 0.5f * width;


	p.y = (1.0f - p.y) * 0.5f * height;
}
void moveCam(camera& cam, float moveSpeed) {
	float xTransform = 0;
	float yTransform = 0;
	float zTransform = 0;

	if (GetAsyncKeyState('W') & 0x8000) {
		zTransform += moveSpeed;
	}

	if (GetAsyncKeyState('A') & 0x8000) {
		xTransform += moveSpeed;
	}

	if (GetAsyncKeyState('S') & 0x8000) {
		zTransform -= moveSpeed;
	}

	if (GetAsyncKeyState('D') & 0x8000) {
		xTransform -= moveSpeed;
	}

	if (GetAsyncKeyState(VK_SPACE) & 0x8000) {
		yTransform -= moveSpeed;
	}
	if (GetAsyncKeyState(VK_LSHIFT) & 0x8000) {
		yTransform += moveSpeed;
	}
	cam.calculateViewMatrix(xTransform, yTransform, zTransform);
}
void rotateCam(camera& cam, float rotateSpeed) {
	float pitch = 0; // Rotation around the X-axis
	float yaw = 0;   // Rotation around the Y-axis

	// Check arrow keys and adjust rotation angles
	if (GetAsyncKeyState(VK_UP) & 0x8000) {
		pitch -= rotateSpeed;
	}
	if (GetAsyncKeyState(VK_DOWN) & 0x8000) {
		pitch += rotateSpeed;
	}
	if (GetAsyncKeyState(VK_LEFT) & 0x8000) {
		yaw -= rotateSpeed;
	}
	if (GetAsyncKeyState(VK_RIGHT) & 0x8000) {
		yaw += rotateSpeed;
	}

	// Apply the rotation to the camera
	// Assuming you have a function like rotateViewMatrix(cam.viewMatrix, pitch, yaw, 0) implemented
	cam.rotateViewMatrix( pitch, yaw, 0);
}

void initDraw(mesh &Mesh) {
	
	int fullSets = Mesh.vertexList.size() / Mesh.batchSize;
	for (int i = 0; i < fullSets; i++) {
		cudaMallocHost((void**)&Mesh.d_matrix, sizeof(float) * 16);
		cudaMallocHost((void**)&Mesh.d_vector, sizeof(float) * Mesh.batchSize * 12);
		cudaMallocHost((void**)&Mesh.d_result, sizeof(float) * Mesh.batchSize * 12);
		Mesh.mData.M_data.push_back(Mesh.d_matrix);
		Mesh.mData.V_data.push_back(Mesh.d_vector);
		Mesh.mData.R_data.push_back(Mesh.d_result);
	}
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
void DrawMesh(HDC hdc, mesh& Mesh, COLORREF color, double width, double height, camera& cam, ThreadPool* pool) {
	std::vector<std::future<std::vector<std::array<POINT, 3>>>> futures;
	std::vector<std::vector<std::array<POINT, 3>>> allFixedPoints;
	std::mutex allFixedPointsMutex;
	int batchSizeL;
	int fullSets;
	if (Mesh.batchSize > Mesh.vertexList.size()) {
		batchSizeL = Mesh.vertexList.size();
		fullSets = Mesh.vertexList.size() / Mesh.batchSize;
	}
	else {
		batchSizeL = Mesh.batchSize;
		fullSets = Mesh.vertexList.size() / Mesh.batchSize;
	}
	if (pool == nullptr) {
		pool = createThreadPool(1);
	}

	for (int i = 0; i < fullSets; ++i) {
		futures.push_back(pool->enqueue([&Mesh, i, batchSizeL, width, height, &cam]() {
			
			std::vector<triangle> triangleBatch(Mesh.vertexList.begin() + i * batchSizeL, Mesh.vertexList.begin() + (i * batchSizeL) + batchSizeL);
			std::vector<std::array<POINT, 3>> localFixedPoints;
			triangleWrapper(Mesh,triangleBatch, width, height, cam, localFixedPoints, i);
			return localFixedPoints; // Return the localFixedPoints directly
			}));
	}

	// Collecting results after all futures have completed
	allFixedPoints.reserve(futures.size()); // Optional: Reserve space in advance

	for (auto& future : futures) {
		allFixedPoints.push_back(future.get());
	}

	// Process any remaining triangles
	int remainder = Mesh.vertexList.size() % batchSizeL;
	if (remainder != 0) {
		std::vector<triangle> triangleBatch(Mesh.vertexList.end() - remainder, Mesh.vertexList.end());
		std::vector<std::array<POINT, 3>> localFixedPoints;
		triangleWrapper(Mesh,triangleBatch, width, height, cam, localFixedPoints,0);
		allFixedPoints.push_back(std::move(localFixedPoints));
	}

	// Draw all triangles on the main thread
	for (auto& fixedPointsBatch : allFixedPoints) {
		for (auto& points : fixedPointsBatch) {
			DrawTriangle(hdc, points, color);
		}
	}
}
mesh loadOBJ(const std::string& filename) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		return  mesh();
	}

	mesh resultMesh;
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
			resultMesh.vertexList.emplace_back(vertices[idx1 - 1], vertices[idx2 - 1], vertices[idx3 - 1]);
		}

	}

	return resultMesh;
}




