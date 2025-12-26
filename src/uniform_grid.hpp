#pragma once 

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <vector>
#include <qem_mesh.hpp>

template <float split_num>
class UniformGrid {
    using m = QEMMesh;

    struct Cell {
        std::vector<m::VertexHandle> vertices;
        std::vector<m::EdgeHandle> edges;
        std::vector<m::FaceHandle> faces;
    };

    std::vector<Cell> cells_;
    float split_ = split_num;

    // Bounding Box
    Eigen::Vector3f min_coords_;
    Eigen::Vector3f max_coords_;


public:

    UniformGrid(Eigen::Vector3f min_coords, Eigen::Vector3f max_coords) : 
        min_coords_(min_coords), max_coords_(max_coords) {};

    size_t add_vertex(const m& mesh, m::VertexHandle vtx, const bool add = true) {
        auto coords = mesh.point(vtx);
        coords[0] -= min_coords_.x();
        coords[1] -= min_coords_.y();
        coords[2] -= min_coords_.z();

        Eigen::Vector3f block_size = (max_coords_ - min_coords_) / split_;
        size_t x = static_cast<size_t>(std::floor(coords[0] / block_size.x()));
        size_t y = static_cast<size_t>(std::floor(coords[0] / block_size.y()));
        size_t z = static_cast<size_t>(std::floor(coords[0] / block_size.z()));
        size_t index = x + (y * split_) + (z * split_ * split_);

        if (add)
            cells_[index].vertices.push_back(vtx);

        return index;
    }

    void add_edge(const m& mesh, m::EdgeHandle eh) {
        auto heh = mesh.halfedge_handle(eh);
        auto vh1 = mesh.from_vertex_handle(heh);
        auto vh2 = mesh.to_vertex_handle(heh);
        size_t idx1 = add_vertex(mesh, vh1, false);
        size_t idx2 = add_vertex(mesh, vh2, false);

        cells_[idx1].edges.push_back(eh);
        if (idx1 != idx2)
            cells_[idx2].edges.push_back(eh);
    }
 
    void add_face(m& mesh, m::FaceHandle fh) {
        std::array<m::VertexHandle, 3> vhs;
        int i = 0;
        for (auto fv_iter = mesh.fv_iter(fh); fv_iter.is_valid(); fv_iter++, i++) {
            auto vh = *fv_iter;
            vhs[i++] = vh;
        }

        size_t idx1 = add_vertex(mesh, vhs[0], false);
        size_t idx2 = add_vertex(mesh, vhs[1], false);
        size_t idx3 = add_vertex(mesh, vhs[2], false);
        
        cells_[idx1].faces.push_back(fh);

        if (idx1 != idx2)
            cells_[idx2].faces.push_back(fh);

        if (idx1 != idx3 && idx2 != idx3)
            cells_[idx3].faces.push_back(fh);
    }
};
