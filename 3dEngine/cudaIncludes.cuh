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



struct point {
public:
	gmtl::Vec4f Position;
	point(float x, float y, float z) {
		Position[0] = x;
		Position[1] = y;
		Position[2] = z;
		Position[3] = 0.0f;

	}

};
struct point2D {
public:
	float x, y;

	point2D(float x, float y) : x(x), y(y) {}
};

//inherits from point because a camera is bassically just a point in space
class camera : public point {
public:
	float fov;
	float nearPlane;
	float farPlane;
	float aspectRatio; // Typically window width / height
	gmtl::Matrix44f projectionMatrix;
	std::string name;
	//rotation stuff

	camera(float x, float y, float z, std::string name, float fov, float aspectRatio, float nearPlane, float farPlane)
		: point(x, y, z), fov(fov), aspectRatio(aspectRatio), nearPlane(nearPlane), farPlane(farPlane), name(name) {
		calculateProjectionMatrix();
	}

	void calculateProjectionMatrix() {
		float yScale = 1.0 / tan((fov / 2.0f) * (M_PI / 180.0f));
		float xScale = yScale / aspectRatio;
		float frustumLength = farPlane - nearPlane;

		gmtl::Matrix44f proj;
		proj.set(xScale, 0, 0, 0,
			0, yScale, 0, 0,
			0, 0, -((farPlane + nearPlane) / frustumLength), -1,
			0, 0, -((2 * nearPlane * farPlane) / frustumLength), 0);
		projectionMatrix = proj;
	}

};
class triangle {
public:
	point p1, p2, p3;

	triangle(const point point1, const point point2, const point point3)
		: p1(point1), p2(point2), p3(point3) {}

};

struct mesh {
public:
	std::vector<triangle> vertexList;
};
struct cudaData {
public:
	std::vector<float*> M_data;
	std::vector<float*> V_data;
	std::vector<float*> R_data;
};