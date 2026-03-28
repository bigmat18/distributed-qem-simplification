#include <utils.hpp>
#include <cstdint>

#include <qem_mesh.hpp>
#include <uniform_grid_qem.hpp>

namespace qems {
    
uint32_t UniformGridQEM::add_vertex(const QEMMesh& mesh, QEMMesh::VertexHandle vh) {
    massert(!mesh.status(vh).deleted(), "Vertex Deleted");
    massert(mesh.is_valid_handle(vh), "Vertex Handle not valid");

    auto indices = get_vertex_indices(mesh, vh);
    cells_[indices.w()].vertices.push_back(vh);
    return indices.w();
}

bool UniformGridQEM::add_edge(const QEMMesh& mesh, QEMMesh::EdgeHandle eh) {
    massert(!mesh.status(eh).deleted(), "Edge Deleted");
    massert(mesh.is_valid_handle(eh), "Edge Handle not valid");
 
    auto heh = mesh.halfedge_handle(eh);
    auto vh1 = mesh.from_vertex_handle(heh);
    auto vh2 = mesh.to_vertex_handle(heh);

    if (mesh.data(vh1).Collasable && mesh.data(vh2).Collasable) {
        uint32_t idx = get_vertex_indices(mesh, vh1).w(); 
        cells_[idx].edges.push_back(eh);
        return true;
    }
    return false;
}

bool UniformGridQEM::increment_collasable_faces(const QEMMesh& mesh, QEMMesh::FaceHandle fh) {
    massert(!mesh.status(fh).deleted(), "Face Deleted");
    massert(mesh.is_valid_handle(fh), "Face Handle not valid");

    for (auto fv_it = mesh.cfv_iter(fh); fv_it.is_valid(); ++fv_it) {
        auto vh = *fv_it;
        if (!mesh.data(vh).Collasable)
            return false;
    }

    auto vh = *mesh.cfv_iter(fh);
    uint32_t idx = get_vertex_indices(mesh, vh).w();
    cells_[idx].collasable_faces++;
    total_collasable_faces_++;
    return true;
}

void UniformGridQEM::merge(const UniformGridQEM& other) {
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

