#pragma once 

#include <cstdint>
#include <string>
#include <filesystem>
#include <vector>

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
                 const std::vector<uint32_t>& faces) {}
}
