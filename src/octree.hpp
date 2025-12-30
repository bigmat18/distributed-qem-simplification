#pragma once 

#include <cmath>
#include <cstddef>
#include <Eigen/Dense>
#include <cstdint>

#include "Eigen/src/Core/Matrix.h"
#include "massert.hpp"
#include "qem_mesh.hpp"

class Octree {
    using Vec4ui = Eigen::Matrix<uint32_t, 4, 1>;
    using Mesh = QEMMesh;

    struct Node {
        bool is_leaf = true;
        size_t collasable_faces = 0;
        std::vector<Mesh::VertexHandle> vertices;
        std::vector<Mesh::EdgeHandle> edges;
    };
    std::vector<Node> tree_;

    // Bounding Box
    Eigen::Vector3f min_coords_;
    Eigen::Vector3f max_coords_;
public:
    Octree(Eigen::Vector3f min_coords, Eigen::Vector3f max_coords) :
        min_coords_(min_coords), max_coords_(max_coords), tree_(8)
    {}

    Octree(const Octree& other) : tree_(8)
    {
        min_coords_ = other.min_coords_;
        max_coords_ = other.max_coords_;
    }

    std::size_t add_vertex(const Mesh& mesh, const Mesh::VertexHandle vh) {
        Eigen::Vector3f max = max_coords_;
        Eigen::Vector3f min = min_coords_;
        std::size_t offset = 0;

        while (true) {
            auto indices = get_vertex_indices(mesh, vh, max, min);
            auto idx = offset + indices.w();
            Node& node = tree_[idx];

            if (node.is_leaf) {
                node.vertices.push_back(vh);
                return idx;
            } else {
                std::size_t jump = 8;
                for (int i = 0; i < indices.w(); i++)
                    if (!tree_[offset + i].is_leaf)
                        jump += 8;

                offset += jump;
                Eigen::Vector3f half = (max - min) * 0.5f;

                if (indices.x() == 0) max.x() = min.x() + half.x();
                else                  min.x() = min.x() + half.x();
                
                if (indices.y() == 0) max.y() = min.y() + half.y();
                else                  min.y() = min.y() + half.y();
                
                if (indices.z() == 0) max.z() = min.z() + half.z();
                else                  min.z() = min.z() + half.z();
            }
        }
    }

    void add_edge(const Mesh& mesh, const Mesh::EdgeHandle eh) { 
        auto heh = mesh.halfedge_handle(eh);
        auto vh1 = mesh.from_vertex_handle(heh);
        auto vh2 = mesh.to_vertex_handle(heh);
        
        if (mesh.data(vh1).Collasable && mesh.data(vh2).Collasable) {
            auto idx1 = mesh.data(vh1).NodeIdx;
            auto idx2 = mesh.data(vh2).NodeIdx;
            massert(idx1 == idx2,"Vertex in differents node both collasable are not allowed");
            tree_[mesh.data(vh1).NodeIdx].edges.push_back(eh);
        }
    }

    void increment_collasable_faces(const Mesh& mesh, const Mesh::FaceHandle fh) {
        for (Mesh::ConstFaceVertexIter fv_it = mesh.cfv_iter(fh);
             fv_it->is_valid(); ++fv_it) 
        {
            auto vh = *fv_it;
            if (!mesh.data(vh).Collasable)
                return;
        }

        auto vh = *mesh.cfv_iter(fh);
        tree_[mesh.data(vh).NodeIdx].collasable_faces++;
    }

    void merge(const Octree& other) {
        for (size_t i = 0; i < tree_.size(); i++) {
            if (!tree_[i].is_leaf)
                continue;

            massert(tree_[i].is_leaf == other.tree_[i].is_leaf,
                    "Can not ve merge different octree");

            tree_[i].vertices.insert(
                tree_[i].vertices.end(),
                other.tree_[i].vertices.begin(),
                other.tree_[i].vertices.end()
            );

            tree_[i].edges.insert(
                tree_[i].edges.end(),
                other.tree_[i].edges.begin(),
                other.tree_[i].edges.end()
            );

            tree_[i].collasable_faces += other.tree_[i].collasable_faces;
        }
    }

    void normalize();

private:

    inline Vec4ui get_vertex_indices(const Mesh& mesh, 
                                     const Mesh::VertexHandle vh,
                                     const Eigen::Vector3f max,
                                     const Eigen::Vector3f min) 
    {
        auto coords = mesh.point(vh);
        coords[0] -= min.x();
        coords[1] -= min.y();
        coords[2] -= min.z();

        Eigen::Vector3f block_size = (max - min) * 0.5f;
        uint32_t x = (coords[0] >= block_size.x()) ? 1 : 0;
        uint32_t y = (coords[1] >= block_size.y()) ? 1 : 0;
        uint32_t z = (coords[2] >= block_size.z()) ? 1 : 0;
        size_t index = x + (y * 2) + (z * 4);

        return Vec4ui(x, y, z, index);
    }
};
