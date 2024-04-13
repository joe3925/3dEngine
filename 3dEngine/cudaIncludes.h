#pragma once
# define M_PI           3.14159265358979323846
#ifndef CUDAINCLUDES_CUH
#define CUDAINCLUDES_CUH


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
#include <map>


class ThreadPool;

struct point {
public:
	gmtl::Vec4f Position;
	point() {

	}
	point(float x, float y, float z) {
		Position[0] = x;
		Position[1] = y;
		Position[2] = z;
	}


};
struct point2D {
public:
	float x, y;

	point2D(float x, float y) : x(x), y(y) {}
	void fixPoint(int width, int height);
};
POINT ConvertFromPoint2D(point2D& pt2D);

class camera : public point {
public:
	float fov;
	float nearPlane;
	float farPlane;
	float aspectRatio;
	point location;
	gmtl::Matrix44f projectionMatrix;
	gmtl::Matrix44f viewMatrix;
	std::string name;
	camera(float x, float y, float z, std::string name, float fov, float aspectRatio, float nearPlane, float farPlane)
		: point(x, y, z), fov(fov), aspectRatio(aspectRatio), nearPlane(nearPlane), farPlane(farPlane), name(name) {
		calculateProjectionMatrix();

		
	}
	camera() {

	}
	void calculateViewMatrix(float x, float y, float z);
	void rotateViewMatrix(float pitch, float yaw, float roll);
	void moveCam(float moveSpeed);
	void rotateCam(float rotateSpeed);
	private:
		gmtl::Matrix44f view;
		void calculateProjectionMatrix();
		void setIdentityMatrix(gmtl::Matrix44f& matrix);

};
class triangle {
public:
	point p1, p2, p3;

	triangle(const point point1, const point point2, const point point3)
		: p1(point1), p2(point2), p3(point3) {}

};


struct cudaData {
public:
	std::vector<float*> PM_data;//proj matrix
	std::vector<float*> VM_data;//view matrix 
	std::vector<float*> VR_data;//view matrix 
	std::vector<float*> V_data;
	std::vector<float*> R_data;
};


struct mesh  {
	mesh() {

	}
	mesh(std::vector<triangle> &triangles) {
		vertexList = triangles;
	}
private:
	float* d_ProjMatrix = nullptr;
	float* d_ViewMatrix = nullptr;
	float* d_vector = nullptr;
	float* d_result = nullptr;
	float* d_ViewResult = nullptr;

	cudaData mData;
	ThreadPool* Meshpool = nullptr;
	int batchSize = 3;
	gmtl::Vec4f translateRotateTranslate(const gmtl::Vec4f& position, const gmtl::Vec4f& center, const gmtl::Matrix44f& rotationMatrix);
	void triangleWrapper(std::vector<triangle>& triangles, float width, float height, camera& cam, std::vector<std::array<POINT, 3>>& fixed, int currentThread);

public:
	std::string Name;
	std::vector<triangle> vertexList;
	std::vector<triangle> deRenderedVertexList;
	boolean init = false;
	int threads = 0;
	std::vector<std::vector<std::array<POINT, 3>>> DrawMesh(HDC hdc, COLORREF color, double width, double height, camera& cam);
	void transform(float x, float y, float z);
	gmtl::Vec4f Center();
	void rotate(float x, float y, float z);
	void setBatchSize(size_t size);
	void initDraw();
	void setPool(ThreadPool* pool);
	bool deRenderRoutine();
};

struct world{
private:
	camera *worldCam = nullptr;
	ThreadPool* pool;
	std::vector<std::string> meshes;
	void initMesh(mesh& Mesh);
	void DrawTriangle(HDC hdc, const std::vector<std::vector<std::array<POINT, 3>>>& pArray, COLORREF color);
public:
	int totalMeshes;
	void renderWorld(HDC hdc, COLORREF color, double width, double height);
	void setThreadPool(ThreadPool* givenPool);
	mesh& addMesh(mesh& Mesh);
	mesh& returnMesh(std::string name);
	mesh& addMeshNotRendered(mesh& Mesh);
	void removeMesh(mesh& mesh);
	void removeMeshByName(std::string name);
	void setCam(camera& cam);
	mesh& deRenderObject(std::string name);
	std::map<std::string, mesh> worldObjects;
	std::map<std::string, mesh> unRenderdObjects;


	


};

#endif