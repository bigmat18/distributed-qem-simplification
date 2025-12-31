#pragma once 

#include <cmath>
#include <cstddef>
#include <Eigen/Dense>
#include <cstdint>
#include <strings.h>
#include <vector>

#include "Eigen/src/Core/Matrix.h"
#include "massert.hpp"
#include "profiling.hpp"
#include "qem_mesh.hpp"

class Octree {
    using Vec4ui = Eigen::Matrix<uint32_t, 4, 1>;
    using BoundingBox = std::pair<Eigen::Vector3f, Eigen::Vector3f>;
    using Mesh = QEMMesh;

    struct Node {
        // Bounding Box
        Eigen::Vector3f min_coords;
        Eigen::Vector3f max_coords;

        bool is_leaf = true;
        size_t collasable_faces = 0;
        std::vector<Mesh::VertexHandle> vertices;
        std::vector<Mesh::EdgeHandle> edges;
    };

    // Bounding Box
    Eigen::Vector3f min_coords_;
    Eigen::Vector3f max_coords_;

    std::vector<Node> tree_;
    std::size_t limit_;

public:
    Octree(Eigen::Vector3f min_coords, Eigen::Vector3f max_coords, size_t limit) :
        min_coords_(min_coords), max_coords_(max_coords), limit_(limit), tree_(8)
    {
        Eigen::Vector3f center = (min_coords + max_coords) * 0.5f;

        for (std::size_t i = 0; i < 8; i++) {
            Node& node = tree_[i];
            BoundingBox bb = get_bounding_box(i, center, min_coords, max_coords);
            node.min_coords = bb.first;
            node.max_coords = bb.second;
        }
    }

    Octree(const Octree& other) : tree_(8)
    {
        min_coords_ = other.min_coords_;
        max_coords_ = other.max_coords_;
        limit_ = other.limit_;

        for (std::size_t i = 0; i < 8; i++) {
            Node& node = tree_[i];
            Node other_node = other.tree_[i];
            node.max_coords = other_node.max_coords;
            node.min_coords = other_node.min_coords;
        }
    }

    std::size_t add_vertex(Mesh& mesh, const Mesh::VertexHandle vh) {
        Eigen::Vector3f max = max_coords_;
        Eigen::Vector3f min = min_coords_;
        std::size_t offset = 0;

        while (true) {
            auto indices = get_vertex_indices(mesh, vh, max, min);
            auto idx = offset + indices.w();
            Node& node = tree_[idx];

            if (node.is_leaf) {
                node.vertices.push_back(vh);
                mesh.data(vh).NodeIdx = idx;
                return idx;
            } else {
                std::size_t jump = 8;
                for (int i = 0; i < indices.w(); i++)
                    if (!tree_[offset + i].is_leaf)
                        jump += 8;

                offset += jump;
                min = node.min_coords;
                max = node.max_coords;
            }
        }
        std::unreachable();
        massert(false, "Unreachable code in add_vertex");
        return -1;
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

    void normalize(Mesh& mesh) {
        PROFILING_SCOPE("Octree-Normalize");

        {
            PROFILING_SCOPE("Reset-Collasable-Faces");
            #pragma omp parallel for schedule(static)
            for(auto& node : tree_) 
                node.collasable_faces = 0;
        }

        for (int i = 0; i < tree_.size(); i++) {
            if (!tree_[i].is_leaf)
                continue;

            if (tree_[i].vertices.size() > limit_)
                split(mesh, i);
        }

        #pragma omp parallel
        {
            PROFILING_SCOPE("Calculate-Collasable-Faces");

            std::vector<uint32_t> local_faces_count(tree_.size(), 0);

            #pragma omp for schedule(static) nowait
            for(size_t i = 0; i < mesh.n_faces(); i++) {
                auto fh = QEMMesh::FaceHandle(i);

                bool collasable = true; 
                for (Mesh::ConstFaceVertexIter fv_it = mesh.cfv_iter(fh);
                     fv_it->is_valid(); ++fv_it) 
                {
                    auto vh = *fv_it;
                    collasable &= mesh.data(vh).Collasable;
                }

                if (collasable) {
                    auto vh = *mesh.cfv_iter(fh);
                    local_faces_count[mesh.data(vh).NodeIdx]++;
                }

            }

            for (size_t i = 0; i < tree_.size(); i++) {
                if (local_faces_count[i] != 0) {
                    #pragma omp atomic
                    tree_[i].collasable_faces += local_faces_count[i];
                }
            }
        }
    }

private:

    void split(Mesh& mesh, int index) {
        std::vector<Mesh::VertexHandle> vertices = std::move(tree_[index].vertices);
        std::vector<Mesh::EdgeHandle> edges = std::move(tree_[index].edges);
        
        Eigen::Vector3f min = tree_[index].min_coords;
        Eigen::Vector3f max = tree_[index].max_coords;
        Eigen::Vector3f center = (min + max) * 0.5f;

        tree_[index].is_leaf = false;
        tree_[index].collasable_faces = 0; 

        size_t first_child_idx = tree_.size();
        tree_.insert(tree_.end(), 8, Node());

        for(size_t i = 0; i < 8; i++) {
            Node& child = tree_[first_child_idx + i];
            BoundingBox bb = get_bounding_box(i, center, min, max);
            child.min_coords = bb.first;
            child.max_coords = bb.second;
        }


        #pragma omp parallel
        {
            std::vector<Mesh::VertexHandle> local_vertices[8];
            std::vector<Mesh::EdgeHandle> local_edges[8];

            #pragma omp for schedule(static) nowait
            for (const auto& vh : vertices) {
                auto indices = get_vertex_indices(mesh, vh, max, min);
                size_t child_local = indices.w(); 
                size_t child_global = first_child_idx + child_local;

                local_vertices[child_local].push_back(vh);
                mesh.data(vh).NodeIdx = child_global;
            } 

            for (int i = 0; i < 8; ++i) {
                if (!local_vertices[i].empty()) {
                    std::size_t idx = first_child_idx + i;
                    #pragma omp critical
                    {
                        tree_[idx].vertices.insert(
                            tree_[idx].vertices.end(),
                            local_vertices[i].begin(),
                            local_vertices[i].end()
                        );
                    }
                }
            }
            #pragma omp barrier

            #pragma omp for schedule(static)
            for (const auto& eh : edges) {    
                auto heh = mesh.halfedge_handle(eh, 0);
                auto vh0 = mesh.from_vertex_handle(heh);
                auto vh1 = mesh.to_vertex_handle(heh);

                size_t idx0 = mesh.data(vh0).NodeIdx;
                size_t idx1 = mesh.data(vh1).NodeIdx;

                if (idx0 != idx1) {
                    mesh.data(vh0).Collasable = false;
                    mesh.data(vh1).Collasable = false;
                }
            }

            #pragma omp for schedule(static) nowait
            for (const auto& eh : edges) {
                auto heh = mesh.halfedge_handle(eh);
                auto vh1 = mesh.from_vertex_handle(heh);
                auto vh2 = mesh.to_vertex_handle(heh);
                
                if (mesh.data(vh1).Collasable && mesh.data(vh2).Collasable) {
                    auto idx1 = mesh.data(vh1).NodeIdx;
                    auto idx2 = mesh.data(vh2).NodeIdx;
                    massert(idx1 == idx2,"Vertex in differents node both collasable are not allowed");

                    size_t local_idx = idx1 - first_child_idx; // 0-7
                    local_edges[local_idx].push_back(eh);
                }
            }

            for (int i = 0; i < 8; ++i) {
                if (!local_edges[i].empty()) {
                    std::size_t idx = first_child_idx + i;
                    #pragma omp critical
                    {

                        tree_[idx].edges.insert(
                            tree_[idx].edges.end(),
                            local_edges[i].begin(),
                            local_edges[i].end()
                        );
                    }
                }
            }
        }
    }

    inline BoundingBox get_bounding_box(int index, 
                                        Eigen::Vector3f center, 
                                        Eigen::Vector3f min, 
                                        Eigen::Vector3f max) 
    {
        BoundingBox result;

        result.first.x() = (index & 1) ? center.x() : min.x();
        result.second.x() = (index & 1) ? max.x() : center.x();

        result.first.y() = (index & 2) ? center.y() : min.y();
        result.second.y() = (index & 2) ? max.y() : center.y();

        result.first.z() = (index & 4) ? center.z() : min.z();
        result.second.z() = (index & 4) ? max.z() : center.z();

        return result;
    }

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
