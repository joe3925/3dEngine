//because of how nvcc is dealing with THpool.h everything with a thread pool needs to be defined here 
#include "cudaIncludes.h"
#include "THpool.h"

std::vector<std::vector<std::array<m_point, 3>>> mesh::DrawMesh(HDC hdc, COLORREF color, double width, double height, camera& cam) {
	std::vector<std::future<std::vector<std::array<m_point, 3>>>> futures;
	std::vector<std::vector<std::array<m_point, 3>>> allFixedPoints;
	std::mutex allFixedPointsMutex;
	int batchSizeL;
	int fullSets;
	if (batchSize > vertexList.size()) {
		batchSizeL = vertexList.size();
		fullSets = vertexList.size() / batchSize;
	}
	else {
		batchSizeL = batchSize;
		fullSets = vertexList.size() / batchSize;
	}
	if (Meshpool == nullptr) {
		for (int i = 0; i < fullSets; ++i) {
			std::vector<triangle> triangleBatch(vertexList.begin() + i * batchSizeL, vertexList.begin() + (i * batchSizeL) + batchSizeL);
			std::vector<std::array<m_point, 3>> localFixedPoints;
			triangleWrapper(triangleBatch, width, height, cam, localFixedPoints, i);
			allFixedPoints.push_back(localFixedPoints);
		}
		return allFixedPoints;
	}

	for (int i = 0; i < fullSets; ++i) {
		futures.push_back(Meshpool->enqueue([this, i, batchSizeL, width, height, &cam]() {

			std::vector<triangle> triangleBatch(vertexList.begin() + i * batchSizeL, vertexList.begin() + (i * batchSizeL) + batchSizeL);
			std::vector<std::array<m_point, 3>> localFixedPoints;
			triangleWrapper(triangleBatch, width, height, cam, localFixedPoints, i);
			return localFixedPoints; // Return the localFixedPoints directly
			}));
	}

	// Collecting results after all futures have completed
	allFixedPoints.reserve(futures.size()); // Optional: Reserve space in advance

	for (auto& future : futures) {
		allFixedPoints.push_back(future.get());
	}

	// Process any remaining triangles
	int remainder = vertexList.size() % batchSizeL;
	if (remainder != 0) {
		std::vector<triangle> triangleBatch(vertexList.end() - remainder, vertexList.end());
		std::vector<std::array<m_point, 3>> localFixedPoints;
		triangleWrapper(triangleBatch, width, height, cam, localFixedPoints, 0);
		allFixedPoints.push_back(std::move(localFixedPoints));
	}

	// Draw all triangles on the main thread
	return allFixedPoints;
}

void mesh::setPool(ThreadPool* pool) {
	Meshpool = pool;
}


void world::renderWorld(HDC hdc, COLORREF color, double width, double height) {
	std::vector<std::future<std::vector<std::vector<std::array<m_point, 3>>>>> futures;
	std::vector<std::vector<std::vector<std::array<m_point, 3>>>> allMeshes;

	if (pool == nullptr) {
		for (int i = 0; i < worldObjects.size(); i++) {
			allMeshes.push_back(worldObjects.at(meshes[i])->DrawMesh(hdc, color, width, height, *worldCam));
		}
		for (auto& _2dMesh : allMeshes) {
			DrawTriangle(hdc, _2dMesh, color);
		}
		return;
	}		
	for (int i = 0; i < totalMeshes; i++) {
		futures.push_back(pool->enqueue([this, i, hdc, color, width, height]()
				{
					return worldObjects.at(meshes[i])->DrawMesh(hdc, color, width, height, *worldCam);

				}));
	}
	allMeshes.reserve(futures.size()); // Optional: Reserve space in advance

	for (auto& future : futures) {
		allMeshes.push_back(future.get());
	}
	for (auto& _2dMesh : allMeshes) {
		DrawTriangle(hdc, _2dMesh, color);
	}

	return;

}
void world::setThreadPool(ThreadPool* givenPool) {
	pool = givenPool;
}

void world::initMesh(mesh *Mesh) {
	if (pool == nullptr) {
		Mesh->threads = 1;
	}
	else {
		Mesh->threads = pool->numThreads;
	}
	Mesh->initDraw();
}