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


#include "cudaIncludes.h"
#include <cudaFile.cuh>


POINT ConvertFromPoint2D(point2D& pt2D)
{
	POINT pt;
	pt.x = static_cast<LONG>(pt2D.x);
	pt.y = static_cast<LONG>(pt2D.y);
	return pt;
}
	
void point2D::fixPoint(int width, int height) {
		x = (x + 1.0f) * 0.5f * width;


		y = (1.0f - y) * 0.5f * height;
}



	void camera::calculateViewMatrix(float x, float y, float z) {

		setIdentityMatrix(view);

		// Translate camera to its position (inversely because moving the world opposite to the camera's movement)
		gmtl::Matrix44f translation;
		gmtl::setTrans(translation, gmtl::Vec3f(-x, -y, -z));

		view = translation * view; // Apply translation

		viewMatrix = view;
		projectionMatrix = projectionMatrix * viewMatrix;
		this->Position[0] += x;
		this->Position[1] += y;
		this->Position[2] += z;


	}
	void camera::rotateViewMatrix(float pitch, float yaw, float roll) {
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
	void camera::moveCam(float moveSpeed) {
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
		calculateViewMatrix(xTransform, yTransform, zTransform);
	}
	void camera::rotateCam(float rotateSpeed) {
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
		rotateViewMatrix(pitch, yaw, 0);
	}


	void camera::calculateProjectionMatrix() {
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
	void camera::setIdentityMatrix(gmtl::Matrix44f& matrix) {
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







	gmtl::Vec4f mesh::translateRotateTranslate(const gmtl::Vec4f& position, const gmtl::Vec4f& center, const gmtl::Matrix44f& rotationMatrix) {
		// Translate to origin
		gmtl::Vec4f translatedPosition = position - center;
		// Rotate
		translatedPosition = rotationMatrix * translatedPosition;
		// Translate back
		translatedPosition += center;
		return translatedPosition;
	}
	void mesh::initDraw() {
		int fullSets = 0;
		if (threads == 0) {
			fullSets = 1;
		}
		else {
			fullSets = vertexList.size() / threads;
		}
		for (int i = 0; i < fullSets; i++) {
			cudaMallocHost((void**)&d_matrix, sizeof(float) * 16);
			cudaMallocHost((void**)&d_vector, sizeof(float) * batchSize * 12);
			cudaMallocHost((void**)&d_result, sizeof(float) * batchSize * 12);
			mData.M_data.push_back(d_matrix);
			mData.V_data.push_back(d_vector);
			mData.R_data.push_back(d_result);
		}
	}
	void mesh::triangleWrapper( std::vector<triangle>& const triangles, float width, float height, camera& cam, std::vector<std::array<POINT, 3>>& fixed, int currentThread) {
		// Calculate the total number of vertices
		int totalVertices = triangles.size() * 3;
		std::vector<point2D> arg3D({ {0,0},{0,0},{0,0},{0,0},{0,0},{0,0} ,{0,0},{0,0},{0,0} }); 

		// Process the triangles' vertices through CUDA
		projectTriangles3Dto2DWithCuda(triangles, cam.projectionMatrix.mData, arg3D, mData.M_data[currentThread], mData.V_data[currentThread], mData.R_data[currentThread]);

		// Iterate over all triangles
		for (size_t t = 0; t < triangles.size(); ++t) {
			// Fix the points for each triangle (-1 - 1 to 0 - width, 0 - height)
			std::array<POINT, 3> points;
			for (int i = 0; i < 3; ++i) {
				arg3D[t * 3 + i].fixPoint(width, height);
				points[i] = ConvertFromPoint2D(arg3D[t * 3 + i]);
			}

			fixed.push_back(points);
		}
	}



	void world::DrawTriangle(HDC hdc, const std::vector<std::vector<std::array<POINT, 3>>>& pArray, COLORREF color) {
		HPEN hPen = CreatePen(PS_SOLID, 1, color);
		HPEN hOldPen = (HPEN)SelectObject(hdc, hPen);
		// Draw the triangle
		for (auto& fixedPointsBatch : pArray) {
			for (auto& points : fixedPointsBatch) {
				Polyline(hdc, points.data(), 3);
			}
		}
		SelectObject(hdc, hOldPen);
		DeleteObject(hPen);
	}

	void mesh::setBatchSize(size_t size) {
		batchSize = size;
	}

	void mesh::transform(float x, float y, float z) {
		for (int i = 0; i < vertexList.size(); i++) {

			vertexList[i].p1.Position[0] += x;
			vertexList[i].p1.Position[1] += y;
			vertexList[i].p1.Position[2] += z;

			vertexList[i].p2.Position[0] += x;
			vertexList[i].p2.Position[1] += y;
			vertexList[i].p2.Position[2] += z;

			vertexList[i].p3.Position[0] += x;
			vertexList[i].p3.Position[0] += y;
			vertexList[i].p3.Position[0] += z;
		}
	}
	gmtl::Vec4f mesh::Center() {
		gmtl::Vec4f center(0.0f, 0.0f, 0.0f, 1.0f);
		for (const triangle& tri : vertexList) {
			center += tri.p1.Position + tri.p2.Position + tri.p3.Position;
		} // Average center position
		return center /= (vertexList.size() * 3);
	}
	void mesh::rotate(float x, float y, float z) {
		// Calculate the center of the mesh
		gmtl::Vec4f center = Center();

		// Create rotation matrices for X, Y, and Z axes
		gmtl::Matrix44f rotationMatrixX, rotationMatrixY, rotationMatrixZ;
		gmtl::setRot(rotationMatrixX, gmtl::AxisAngle<float>(gmtl::Math::deg2Rad(x), 1.0f, 0.0f, 0.0f)); // Rotate around X axis
		gmtl::setRot(rotationMatrixY, gmtl::AxisAngle<float>(gmtl::Math::deg2Rad(y), 0.0f, 1.0f, 0.0f)); // Rotate around Y axis
		gmtl::setRot(rotationMatrixZ, gmtl::AxisAngle<float>(gmtl::Math::deg2Rad(z), 0.0f, 0.0f, 1.0f)); // Rotate around Z axis

		// Combine the rotations (order matters)
		gmtl::Matrix44f combinedRotationMatrix = rotationMatrixZ * rotationMatrixY * rotationMatrixX;

		// move the mesh to the orgin rotate it and move it back
		for (triangle& tri : vertexList) {
			tri.p1.Position = translateRotateTranslate(tri.p1.Position, center, combinedRotationMatrix);
			tri.p2.Position = translateRotateTranslate(tri.p2.Position, center, combinedRotationMatrix);
			tri.p3.Position = translateRotateTranslate(tri.p3.Position, center, combinedRotationMatrix);
		}
	}

	void world::addMesh(mesh &Mesh) {
		worldObjects[Mesh.Name] = Mesh;
		meshes.push_back(Mesh.Name);
		worldObjects[Mesh.Name].setPool(pool);
		initMesh(worldObjects[Mesh.Name]);
		totalMeshes++;
	}
	void world::removeMesh(mesh& Mesh) {
		std::string name = Mesh.Name;
		auto i = std::find(meshes.begin(), meshes.end(), name);
		meshes.erase(meshes.begin(), i);
		worldObjects.erase(Mesh.Name);
		totalMeshes--;
	}
	void world::removeMeshByName(std::string name) {
		auto i = std::find(meshes.begin(), meshes.end(), name);
		meshes.erase(meshes.begin(), i);
		worldObjects.erase(name);
		totalMeshes--;
	}

	void world::setCam(camera& cam) {
		worldCam = cam;
	}


