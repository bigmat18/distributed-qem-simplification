#pragma once 

#include "logging.hpp"
#include <cstdint>
#include <string>
#include <filesystem>
#include <vector>
#include <fstream>

#include <utils.hpp>
#include <Eigen/Dense>

namespace qems {

struct MeshMetaData {
    std::string name;
    std::string type;

    Eigen::Vector3d min_coords;
    Eigen::Vector3d max_coords;
};

void import_mesh(const std::filesystem::path path, 
                 MeshMetaData& metadata,
                 std::vector<float>& vertices,
                 std::vector<uint32_t>& faces);

inline void export_mesh(const std::filesystem::path path,
                        const std::vector<float>& vertices,
                        const std::vector<uint32_t>& faces) 
{
    std::ofstream file(path);
    massert(file.is_open(), "Error: Failed to open PLY file.");

    const size_t num_vertices = vertices.size() / 3;
    const size_t num_faces = faces.size() / 3;

    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << num_vertices << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "element face " << num_faces << "\n";
    file << "property list uchar int vertex_indices\n"; 
    file << "end_header\n";

    for (size_t i = 0; i < vertices.size(); i += 3) {
        file << vertices[i] << " " 
             << vertices[i + 1] << " " 
             << vertices[i + 2] << "\n";
    }

    for (size_t i = 0; i < faces.size(); i += 3) {
        file << "3 " << faces[i] << " " 
             << faces[i + 1] << " " 
             << faces[i + 2] << "\n";
    }

    file.close();
}

}
