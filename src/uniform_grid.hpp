#pragma once 

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <vector>
#include <qem_mesh.hpp>


template <uint32_t split_num>
class UniformGrid {
    using m = QEMMesh;

    struct Cell {
        std::vector<m::VertexHandle> vertices;
        std::vector<m::EdgeHandle> edges;
        std::vector<m::FaceHandle> faces;
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

    size_t add_vertex(m& mesh, m::VertexHandle vh, const bool add = true) {
        auto coords = mesh.point(vh);
        coords[0] -= min_coords_.x();
        coords[1] -= min_coords_.y();
        coords[2] -= min_coords_.z();

        Eigen::Vector3f block_size = (max_coords_ - min_coords_) / split_;
        size_t x = std::min(static_cast<size_t>(std::floor(coords[0] / block_size.x())), static_cast<size_t>(split_ - 1));
        size_t y = std::min(static_cast<size_t>(std::floor(coords[1] / block_size.y())), static_cast<size_t>(split_ - 1));
        size_t z = std::min(static_cast<size_t>(std::floor(coords[2] / block_size.z())), static_cast<size_t>(split_ - 1));
        size_t index = x + (y * split_) + (z * split_ * split_);

        if (add) {
            cells_[index].vertices.push_back(vh);
            mesh.set_color(
                vh,
                m::Color(
                    static_cast<unsigned char>(x * ((256 / split_) - 1)),
                    static_cast<unsigned char>(y * ((256 / split_) - 1)),
                    static_cast<unsigned char>(z * ((256 / split_) - 1))));
        }

        return index;
    }

    void add_edge(m& mesh, m::EdgeHandle eh) {
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
        for (auto fv_iter = mesh.fv_iter(fh); fv_iter.is_valid(); fv_iter++) {
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
 
            cells_[i].faces.insert(
                cells_[i].faces.end(),
                mesh.cells_[i].faces.begin(),
                mesh.cells_[i].faces.end()
            );
        }
    }
};
