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
	point() {

	}
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
	void calculateViewMatrix(float x, float y, float z) {

		setIdentityMatrix(view);

		// Translate camera to its position (inversely because moving the world opposite to the camera's movement)
		gmtl::Matrix44f translation;
		gmtl::setTrans(translation, gmtl::Vec3f(-x, -y, -z));
		
		view = translation * view; // Apply translation

		// Here you should apply rotation based on the camera's orientation
		// For example, if you have Euler angles or a quaternion representing the camera's rotation,
		// you would construct a rotation matrix and multiply it with the view matrix here

		viewMatrix = view;
		projectionMatrix = projectionMatrix * viewMatrix;
		this->Position[0] += x;
		this->Position[1] += y;
		this->Position[2] += z;

		
	}
	void rotateViewMatrix(float pitch, float yaw, float roll) {
		// Convert angles from degrees to radians for trigonometric functions
		float pitchRad = pitch * (M_PI / 180.0f);
		float yawRad = yaw * (M_PI / 180.0f);
		float rollRad = roll * (M_PI / 180.0f);

		// Create rotation matrices around the x, y, and z axes
		gmtl::Matrix44f rotX, rotY, rotZ;
		setIdentityMatrix(rotX);
		setIdentityMatrix(rotY);
		setIdentityMatrix(rotZ);

		// Rotation matrix for pitch (X-axis)
		rotX(1, 1) = cos(pitchRad);
		rotX(1, 2) = -sin(pitchRad);
		rotX(2, 1) = sin(pitchRad);
		rotX(2, 2) = cos(pitchRad);

		// Rotation matrix for yaw (Y-axis)
		rotY(0, 0) = cos(yawRad);
		rotY(0, 2) = sin(yawRad);
		rotY(2, 0) = -sin(yawRad);
		rotY(2, 2) = cos(yawRad);

		// Rotation matrix for roll (Z-axis)
		rotZ(0, 0) = cos(rollRad);
		rotZ(0, 1) = -sin(rollRad);
		rotZ(1, 0) = sin(rollRad);
		rotZ(1, 1) = cos(rollRad);

		// Combine rotations: first roll, then pitch, then yaw
		// The order of multiplication is important and depends on how you define the axes and rotation order
		gmtl::Matrix44f rotation = rotY * rotX * rotZ;

		// Apply rotation to the view matrix
		viewMatrix = rotation * viewMatrix;
		projectionMatrix = projectionMatrix * viewMatrix;
	}

	private:
		gmtl::Matrix44f view;
		void setIdentityMatrix(gmtl::Matrix44f& matrix) {
			// Reset all elements to 0
			for (int row = 0; row < 4; ++row) {
				for (int col = 0; col < 4; ++col) {
					matrix(row, col) = 0.0f;
				}
			}

			// Set diagonal elements to 1
			matrix(0, 0) = 1.0f;
			matrix(1, 1) = 1.0f;
			matrix(2, 2) = 1.0f;
			matrix(3, 3) = 1.0f;
		}

};
class triangle {
public:
	point p1, p2, p3;

	triangle(const point point1, const point point2, const point point3)
		: p1(point1), p2(point2), p3(point3) {}

};
struct cudaData {
public:
	std::vector<float*> M_data;
	std::vector<float*> V_data;
	std::vector<float*> R_data;
};
struct mesh {
public:
	std::vector<triangle> vertexList;
	float* d_matrix;
	float* d_vector;
	float* d_result;
	cudaData mData;
	int batchSize = 3;
};
