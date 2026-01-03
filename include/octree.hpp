#pragma once

#include <cmath>
#include <cstddef>
#include <Eigen/Dense>
#include <cstdint>
#include <filesystem>
#include <strings.h>
#include <vector>

#include "qem_mesh.hpp"

namespace qems {

class Octree { 
    struct Node {
        // Bounding Box
        Eigen::Vector3d min_coords;
        Eigen::Vector3d max_coords;

        bool is_leaf = true;
        uint32_t collasable_faces = 0;
        std::vector<QEMMesh::VertexHandle> vertices;
        std::vector<QEMMesh::EdgeHandle> edges;
    };

    using Vec4ui = Eigen::Matrix<uint32_t, 4, 1>;
    using BoundingBox = std::pair<Eigen::Vector3d, Eigen::Vector3d>;
    using BalancedPartitions = std::vector<std::vector<const Node*>>;

    // Bounding Box
    Eigen::Vector3d min_coords_;
    Eigen::Vector3d max_coords_;

    std::vector<Node> tree_;
    std::size_t limit_ = 1000;
    uint32_t total_collasable_faces_ = 0;

public:
    Octree() = default;


    Octree(Eigen::Vector3d min_coords, Eigen::Vector3d max_coords, size_t limit) :
        min_coords_(min_coords), max_coords_(max_coords), limit_(limit), tree_(8)
    {
        Eigen::Vector3d center = (min_coords + max_coords) * 0.5f;

        for (std::size_t i = 0; i < 8; i++) {
            Node& node = tree_[i];
            BoundingBox bb = compute_bounding_box(i, center, min_coords, max_coords);
            node.min_coords = bb.first;
            node.max_coords = bb.second;
        }
    }


    Octree(const Octree& other) : 
        min_coords_(other.min_coords_), max_coords_(other.max_coords_), 
        limit_(other.limit_), tree_(8)
    {
        for (std::size_t i = 0; i < 8; i++) {
            Node& node = tree_[i];
            Node other_node = other.tree_[i];
            node.max_coords = other_node.max_coords;
            node.min_coords = other_node.min_coords;
        }
    }

    uint32_t add_vertex(const QEMMesh& mesh, const QEMMesh::VertexHandle vh);

    void add_edge(const QEMMesh& mesh, const QEMMesh::EdgeHandle eh);

    void increment_collasable_faces(const QEMMesh& mesh, const QEMMesh::FaceHandle fh); 

    void merge(const Octree& other);

    void normalize(QEMMesh& mesh);

    void export_mesh(const std::filesystem::path path);

    inline uint32_t total_collasable_faces() const { return total_collasable_faces_; }

    inline Vec4ui get_vertex_indices(const QEMMesh& mesh, 
                                     const QEMMesh::VertexHandle vh,
                                     const Eigen::Vector3d min,
                                     const Eigen::Vector3d max) const
    {
        auto coords = mesh.point(vh);
        coords[0] -= min.x();
        coords[1] -= min.y();
        coords[2] -= min.z();

        Eigen::Vector3d block_size = (max - min) * 0.5f;
        uint32_t x = (coords[0] >= block_size.x()) ? 1 : 0;
        uint32_t y = (coords[1] >= block_size.y()) ? 1 : 0;
        uint32_t z = (coords[2] >= block_size.z()) ? 1 : 0;
        size_t index = x + (y * 2) + (z * 4);

        return Vec4ui(x, y, z, index);
    }

    inline std::vector<Node>& get_nodes() { return tree_; }

    inline BalancedPartitions get_balanced_partitioned_tree(uint32_t partitions) {
        std::vector<std::vector<const Node*>> result;
        
        return result;
    }

private:

    void split(QEMMesh& mesh, uint32_t idx);

    inline BoundingBox compute_bounding_box(std::size_t idx, 
                                            Eigen::Vector3d center,
                                            Eigen::Vector3d min,
                                            Eigen::Vector3d max) 
    {
        BoundingBox result;

        result.first.x() = (idx & 1) ? center.x() : min.x();
        result.second.x() = (idx & 1) ? max.x() : center.x();

        result.first.y() = (idx & 2) ? center.y() : min.y();
        result.second.y() = (idx & 2) ? max.y() : center.y();

        result.first.z() = (idx & 4) ? center.z() : min.z();
        result.second.z() = (idx & 4) ? max.z() : center.z();

        return result;
    }
};

}
