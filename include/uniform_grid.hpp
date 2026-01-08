#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <qem_mesh.hpp>

namespace qems {

class UniformGrid {
    using Vec4ui = Eigen::Matrix<uint32_t, 4, 1>;

    struct Cell {
        uint32_t collasable_faces = 0;
        std::vector<QEMMesh::VertexHandle> vertices;
        std::vector<QEMMesh::EdgeHandle> edges;
    };

    // Bounding Box
    Eigen::Vector3d min_coords_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d max_coords_ = Eigen::Vector3d::Zero();

    uint32_t total_collasable_faces_ = 0;
    uint32_t num_split_ = 4;

    std::vector<Cell> cells_;
public:
    using value_type      = Cell;
    using iterator        = std::vector<Cell>::iterator;
    using const_iterator  = std::vector<Cell>::const_iterator;

    UniformGrid() = default;

    UniformGrid(Eigen::Vector3d min_coords, Eigen::Vector3d max_coords, uint32_t num_split = 4) : 
        min_coords_(min_coords), max_coords_(max_coords), num_split_(num_split),
        cells_(num_split_ * num_split_ * num_split_) {};

    UniformGrid(const UniformGrid& other) :
        min_coords_(other.min_coords_), max_coords_(other.max_coords_), 
        num_split_(other.num_split_),
        cells_(num_split_ * num_split_ * num_split_) {};

    uint32_t add_vertex(const QEMMesh& mesh, QEMMesh::VertexHandle vh);

    bool add_edge(const QEMMesh& mesh, QEMMesh::EdgeHandle eh);

    bool increment_collasable_faces(const QEMMesh& mesh, QEMMesh::FaceHandle fh);

    void merge(const UniformGrid& other);

    inline uint32_t total_collasable_faces() const { return total_collasable_faces_; }

    iterator begin() { return cells_.begin(); }
    iterator end()   { return cells_.end(); }

    const_iterator begin()  const { return cells_.begin(); }
    const_iterator end()    const { return cells_.end(); }
    const_iterator cbegin() const { return cells_.cbegin(); }
    const_iterator cend()   const { return cells_.cend(); }

    std::size_t size() const { return cells_.size(); }

private:
    inline Vec4ui get_vertex_indices(const QEMMesh& mesh, QEMMesh::VertexHandle vh) {
        auto coords = mesh.point(vh);
        coords[0] -= min_coords_.x();
        coords[1] -= min_coords_.y();
        coords[2] -= min_coords_.z();

        Eigen::Vector3d block_size = (max_coords_ - min_coords_) / num_split_;
        uint32_t x = std::min(static_cast<uint32_t>(std::floor(coords[0] / block_size.x())), num_split_ - 1);
        uint32_t y = std::min(static_cast<uint32_t>(std::floor(coords[1] / block_size.y())), num_split_ - 1);
        uint32_t z = std::min(static_cast<uint32_t>(std::floor(coords[2] / block_size.z())), num_split_ - 1);
        uint32_t index = x + (y * num_split_) + (z * num_split_ * num_split_);

        return Vec4ui(x, y, z, index);
    }
};
}
