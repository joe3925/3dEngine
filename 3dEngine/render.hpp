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






void DrawTriangle(HDC hdc, triangle Triangle, COLORREF color, double width, double height, const camera& cam);
void DrawMesh(HDC hdc, mesh& Mesh, COLORREF color, double width, double height, const camera& cam);
POINT ConvertFromPoint2D(point2D& pt2D);
void fixPoint(point2D &p, int width, int height);
point2D Project3Dto2D(const point& pt3D, const camera& cam);
gmtl::Vec4f translateRotateTranslate(const gmtl::Vec4f& position, const gmtl::Vec4f& center, const gmtl::Matrix44f& rotationMatrix);


void DrawTriangle(HDC hdc, triangle Triangle, COLORREF color, double width, double height, const camera& cam) {


	point2D p1 = Project3Dto2D(Triangle.p1, cam);
	point2D p2 = Project3Dto2D(Triangle.p2, cam);
	point2D p3 = Project3Dto2D(Triangle.p3, cam);
	HPEN hPen = CreatePen(PS_SOLID, 1, color);
	HPEN hOldPen = (HPEN)SelectObject(hdc, hPen);
	
	//fix the points (-1 - 1 to 0 - width, 0 - height)
	fixPoint(p1, width,height);
	fixPoint(p2, width, height);
	fixPoint(p3, width, height);
	POINT trianglePoints[4] = { ConvertFromPoint2D(p1), ConvertFromPoint2D(p2), ConvertFromPoint2D(p3), ConvertFromPoint2D(p1) };

	// Draw the triangle
	Polyline(hdc, trianglePoints, 4); 

	SelectObject(hdc, hOldPen);
	DeleteObject(hPen);
}
void DrawMesh(HDC hdc, mesh &Mesh, COLORREF color, double width, double height, const camera& cam) {
	for (int i = 0; i < Mesh.vertexList.size(); i++) {
		DrawTriangle(hdc, Mesh.vertexList[i], color, width, height, cam);
	}
	
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


void transform(mesh &Mesh, float x, float y, float z) {
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



point2D Project3Dto2D(const point& pt3D, const camera& cam) {
	// Translate the point relative to the camera position
	gmtl::Vec4f pointRelativeToCamera = pt3D.Position;
	pointRelativeToCamera[0] -= cam.Position[0];
	pointRelativeToCamera[1] -= cam.Position[1];
	pointRelativeToCamera[2] -= cam.Position[2];

	// Apply the projection matrix to the translated point
	gmtl::Vec4f projected = cam.projectionMatrix * pointRelativeToCamera;

	// Perform perspective divide
	if (projected[3] != 0.0f) {
		projected[0] /= projected[3];
		projected[1] /= projected[3];
	}

	float x = projected[0];
	float y = projected[1];

	return point2D(x, y);
}

POINT ConvertFromPoint2D( point2D& pt2D) {
	POINT pt;
	pt.x = static_cast<LONG>(pt2D.x); 
	pt.y = static_cast<LONG>(pt2D.y);
	return pt;
}

void fixPoint(point2D& p, int width, int height) {
	p.x = (p.x + 1.0f) * 0.5f * width;


	p.y = (1.0f - p.y) * 0.5f * height;
}
void moveCam(camera &cam, float moveSpeed) {
	if (GetAsyncKeyState('W') & 0x8000) {
		cam.Position[2] += moveSpeed;
	}

	if (GetAsyncKeyState('A') & 0x8000) {
		cam.Position[0] += moveSpeed;
	}

	if (GetAsyncKeyState('S') & 0x8000) {
		cam.Position[2] -= moveSpeed;
	}

	if (GetAsyncKeyState('D') & 0x8000) {
		cam.Position[0] -= moveSpeed;
	}

	if (GetAsyncKeyState(VK_SPACE) & 0x8000) {
		cam.Position[1] -= moveSpeed;
	}
	if (GetAsyncKeyState(VK_LSHIFT) & 0x8000) {
		cam.Position[1] += moveSpeed;
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