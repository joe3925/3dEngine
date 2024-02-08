#pragma once

#include <gmtl/gmtl.h>
#include <windows.h>
#include <vector>
#include <string>
#include <iostream>
#include <wingdi.h>

struct point {
public:
	gmtl::Vec4f Postition;
	point(float x, float y, float z) {
		Postition[0] = x;
		Postition[1] = y;
		Postition[2] = z;
		Postition[3] = 1.0f;

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
	std::string name;
	camera(float x, float y, float z, std::string name , float fov)
		: point(x, y, z), fov(fov), name(name){
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






void DrawTriangle(HDC hdc, triangle Triangle, COLORREF color, double width, double height);
void DrawMesh(HDC hdc, mesh &Mesh, COLORREF color, double width, double height);

point2D Project3Dto2D(point& pt3D);
POINT ConvertFromPoint2D(point2D& pt2D);
void fixPoint(point2D &p, int width, int height);


void DrawTriangle(HDC hdc, triangle Triangle, COLORREF color, double width, double height) {


	point2D p1 = Project3Dto2D(Triangle.p1);
	point2D p2 = Project3Dto2D(Triangle.p2);
	point2D p3 = Project3Dto2D(Triangle.p3);
	HPEN hPen = CreatePen(PS_SOLID, 1, color);
	HPEN hOldPen = (HPEN)SelectObject(hdc, hPen);
	
	//fix the points (-1 - 1 to 0 - width, 0 - height)
	fixPoint(p1, width,height);
	fixPoint(p2, width, height);
	fixPoint(p3, width, height);
	POINT trianglePoints[3] = { ConvertFromPoint2D(p1), ConvertFromPoint2D(p2), ConvertFromPoint2D(p3)}; 

	// Draw the triangle
	Polyline(hdc, trianglePoints, 3); 

	SelectObject(hdc, hOldPen);
	DeleteObject(hPen);
}
void DrawMesh(HDC hdc, mesh &Mesh, COLORREF color, double width, double height) {
	for (int i = 0; i < Mesh.vertexList.size(); i++) {
		DrawTriangle(hdc, Mesh.vertexList[i], color, width, height);
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
		
		Mesh.vertexList[i].p1.Postition[0] += x;
		Mesh.vertexList[i].p1.Postition[1] += y;
		Mesh.vertexList[i].p1.Postition[2] += z;

		Mesh.vertexList[i].p2.Postition[0] += x;
		Mesh.vertexList[i].p2.Postition[1] += y;
		Mesh.vertexList[i].p2.Postition[2] += z;

		Mesh.vertexList[i].p3.Postition[0] += x;
		Mesh.vertexList[i].p3.Postition[1] += y;
		Mesh.vertexList[i].p3.Postition[2] += z;
	}
}
//TODO: test this
void rotate(mesh& Mesh, float x, float y, float z) {
	// Create rotation matrices for X, Y, and Z axes
	gmtl::Matrix44f rotationMatrixX, rotationMatrixY, rotationMatrixZ;
	gmtl::setRot(rotationMatrixX, gmtl::AxisAngle<float>(gmtl::Math::deg2Rad(x), 1.0f, 0.0f, 0.0f)); // Rotate around X axis
	gmtl::setRot(rotationMatrixY, gmtl::AxisAngle<float>(gmtl::Math::deg2Rad(y), 0.0f, 1.0f, 0.0f)); // Rotate around Y axis
	gmtl::setRot(rotationMatrixZ, gmtl::AxisAngle<float>(gmtl::Math::deg2Rad(z), 0.0f, 0.0f, 1.0f)); // Rotate around Z axis

	// Combine the rotations (order matters )
	gmtl::Matrix44f combinedRotationMatrix = rotationMatrixZ * rotationMatrixY * rotationMatrixX;

	// Apply the combined rotation to each vertex in each triangle of the mesh
	for (triangle& tri : Mesh.vertexList) {
		tri.p1.Postition = combinedRotationMatrix * tri.p1.Postition;
		tri.p2.Postition = combinedRotationMatrix * tri.p2.Postition;
		tri.p3.Postition = combinedRotationMatrix * tri.p3.Postition;
	}
}

point2D Project3Dto2D( point& pt3D) {
	
	float x = pt3D.Postition[0]*(pt3D.Postition[3] / pt3D.Postition[2]);
	float y = pt3D.Postition[1] * (pt3D.Postition[3] / pt3D.Postition[2]);

	return point2D(x, y);
}
POINT ConvertFromPoint2D( point2D& pt2D) {
	POINT pt;
	pt.x = static_cast<LONG>(pt2D.x); 
	pt.y = static_cast<LONG>(pt2D.y);
	return pt;
}

void fixPoint (point2D &p, int width, int height) {
	if (p.x > 0) {
		p.x = p.x * width/2; //get the percent of half the width 
		p.x = p.x + width / 2; // p.x is positive so the coord is on the right side so add width/2
	}
	if (p.x < 0) {
		p.x = (width / 2) - (abs(p.x) * width / 2); //get the percent of half the width 
	}
	if (p.x == 0) {
		p.x = width / 2; //get the percent of half the width 
	}
	
	//do the same for y
	if (p.y > 0) {
		p.y = p.y * height / 2; //get the percent of half the height 
		p.y = p.y + height / 2; // p.x is positive so the coord is on the right side so add width/2
	}
	if (p.y < 0) {
		p.y= (width / 2) - (abs(p.y) * height / 2); //get the percent of half the height 
	}
	if (p.y == 0) {
		p.y = height / 2; //get the percent of half the width 
	}
}