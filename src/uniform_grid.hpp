#pragma once 

#include "qem_utils.hpp"
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>
#include <qem_mesh.hpp>
    


template <uint32_t split_num>
class UniformGrid {
    using m = QEMMesh;
    using Vec4ui = Eigen::Matrix<uint32_t, 4, 1>;

    struct Cell {
        std::vector<m::VertexHandle> vertices;
        std::vector<m::EdgeHandle> edges;
    };

    std::vector<Cell> cells_;
    uint32_t split_ = split_num;

    // Bounding Box
    Eigen::Vector3f min_coords_;
    Eigen::Vector3f max_coords_;


public:

    UniformGrid(Eigen::Vector3f min_coords, Eigen::Vector3f max_coords) : 
        cells_(split_num * split_num * split_num), min_coords_(min_coords), max_coords_(max_coords) 
    {};

    UniformGrid(const UniformGrid<split_num>& other) :
        cells_(split_num * split_num * split_num)
    {
        min_coords_ = other.min_coords_;
        max_coords_ = other.max_coords_;
    }

    Vec4ui get_vertex_indices(const m& mesh, m::VertexHandle vh) {
        auto coords = mesh.point(vh);
        coords[0] -= min_coords_.x();
        coords[1] -= min_coords_.y();
        coords[2] -= min_coords_.z();

        Eigen::Vector3f block_size = (max_coords_ - min_coords_) / split_;
        size_t x = std::min(static_cast<size_t>(std::floor(coords[0] / block_size.x())), static_cast<size_t>(split_ - 1));
        size_t y = std::min(static_cast<size_t>(std::floor(coords[1] / block_size.y())), static_cast<size_t>(split_ - 1));
        size_t z = std::min(static_cast<size_t>(std::floor(coords[2] / block_size.z())), static_cast<size_t>(split_ - 1));
        size_t index = x + (y * split_) + (z * split_ * split_);

        return Vec4ui(x, y, z, index);
    }

    void add_vertex(m& mesh, m::VertexHandle vh) {
        auto indices = get_vertex_indices(mesh, vh);

        cells_[indices.w()].vertices.push_back(vh);
        mesh.set_color(
            vh,
            m::Color(
                static_cast<unsigned char>(indices.x() * ((256 / split_) - 1)),
                static_cast<unsigned char>(indices.y() * ((256 / split_) - 1)),
                static_cast<unsigned char>(indices.z() * ((256 / split_) - 1))));
    }

    void add_edge(m& mesh, m::EdgeHandle eh) {
        auto heh = mesh.halfedge_handle(eh);
        auto vh1 = mesh.from_vertex_handle(heh);
        auto vh2 = mesh.to_vertex_handle(heh);
        size_t idx1 = get_vertex_indices(mesh, vh1).w(); 
        size_t idx2 = get_vertex_indices(mesh, vh2).w(); 
            
        if (mesh.data(vh1).Collapable && mesh.data(vh2).Collapable)
            cells_[idx1].edges.push_back(eh);
    } 

    void merge(const UniformGrid<split_num>& mesh) {
        for (int i = 0; i < cells_.size(); ++i) {
            cells_[i].vertices.insert(
                cells_[i].vertices.end(),
                mesh.cells_[i].vertices.begin(),
                mesh.cells_[i].vertices.end()
            );

            cells_[i].edges.insert(
                cells_[i].edges.end(),
                mesh.cells_[i].edges.begin(),
                mesh.cells_[i].edges.end()
            );
        }
    }

    QEMPriorityQueue get_qem_pq(const m& mesh, size_t index) { 
        return QEMPriorityQueue(QEMEdgeCompare(&mesh), cells_[index].edges);
    }
};
