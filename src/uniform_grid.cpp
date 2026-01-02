#include "qem_mesh.hpp"
#include <utils.hpp>
#include <cstdint>
#include <uniform_grid.hpp>

namespace qems {
    
void UniformGrid::add_vertex(const QEMMesh& mesh, QEMMesh::VertexHandle vh) {
    if (mesh.status(vh).deleted() || !mesh.is_valid_handle(vh))
        return;

    auto indices = get_vertex_indices(mesh, vh);
    cells_[indices.w()].vertices.push_back(vh);
}

void UniformGrid::add_edge(const QEMMesh& mesh, QEMMesh::EdgeHandle eh) {
    if (mesh.status(eh).deleted() || !mesh.is_valid_handle(eh))
        return;
        
    auto heh = mesh.halfedge_handle(eh);
    auto vh1 = mesh.from_vertex_handle(heh);
    auto vh2 = mesh.to_vertex_handle(heh);

    if (mesh.data(vh1).Collasable && mesh.data(vh2).Collasable) {
        uint32_t idx = get_vertex_indices(mesh, vh1).w(); 
        cells_[idx].edges.push_back(eh);
    }
}

void UniformGrid::increment_collasable_faces(const QEMMesh& mesh, QEMMesh::FaceHandle fh) {
    if (mesh.status(fh).deleted() || !mesh.is_valid_handle(fh))
        return;

    for (auto fv_it = mesh.cfv_iter(fh); fv_it.is_valid(); ++fv_it) {
        auto vh = *fv_it;
        if (!mesh.data(vh).Collasable)
            return;
    }

    auto vh = *mesh.cfv_iter(fh);
    uint32_t idx = get_vertex_indices(mesh, vh).w();
    cells_[idx].collasable_faces++;
    total_collasable_faces_++;
}

void UniformGrid::merge(const UniformGrid& other) {
    for (int i = 0; i < cells_.size(); ++i) {
        cells_[i].vertices.insert(
            cells_[i].vertices.end(),
            other.cells_[i].vertices.begin(),
            other.cells_[i].vertices.end()
        );

        cells_[i].edges.insert(
            cells_[i].edges.end(),
            other.cells_[i].edges.begin(),
            other.cells_[i].edges.end()
        );

        cells_[i].collasable_faces += other.cells_[i].collasable_faces;
    }
    total_collasable_faces_ += other.total_collasable_faces_;
}

}

