#pragma once 

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "qem_mesh.hpp"

namespace qems {

enum class ImportType {
    ROW_DATA,
    MESH_DATA,
    ROW_MESH_DATA
};


struct MeshData {
    std::string name;
    std::string type;

    std::vector<float> row_vertices;
    std::vector<uint32_t> row_faces;

    Eigen::Vector3d min_coords;
    Eigen::Vector3d max_coords;

    QEMMesh mesh;
};

namespace detail {

template <ImportType type>
void import_ply(const std::filesystem::path path, MeshData& data);

template <ImportType type>
void import_obj(const std::filesystem::path path, MeshData& data);

}

template <ImportType type>
void import_mesh(const std::filesystem::path path, MeshData& data);

}
