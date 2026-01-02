#include <qem_simp.hpp>
#include <qem_mesh.hpp>
#include <cmath>
#include <limits>

namespace qems {

void simplification(QEMMesh &mesh, uint32_t target, uint32_t num_faces, QEMPriorityQueue &pq) {
    uint32_t deleted_faces = 0;
    while (num_faces - deleted_faces > target && !pq.empty()) { 
        auto eh = pq.top();
        pq.pop();

        if (mesh.status(eh).deleted())
            continue;

        auto heh = mesh.halfedge_handle(eh, 0);

        if (!mesh.is_collapse_ok(heh))
            continue;

        auto vh0 = mesh.from_vertex_handle(heh);
        auto vh1 = mesh.to_vertex_handle(heh);

        if (mesh.status(vh0).deleted() || mesh.status(vh1).deleted()) 
            continue;

        Eigen::Vector4d newVertex = mesh.data(eh).NewVertex;
        OpenMesh::Vec3f coords(newVertex.x(), newVertex.y(), newVertex.z());

        mesh.set_point(vh1, coords);
        mesh.data(vh1).Quadric = mesh.data(vh1).Quadric + mesh.data(vh0).Quadric;
        mesh.collapse(heh);

        std::vector<QEMMesh::VertexHandle> vertices;
        std::vector<QEMMesh::EdgeHandle> edges;

        for (auto fh : mesh.vf_range(vh1)) {
            if (mesh.status(fh).deleted()) continue;

            for (auto vh : mesh.fv_range(fh)) {
                if (mesh.status(vh).deleted() || !mesh.data(vh).Collasable) 
                    continue;

                vertices.push_back(vh);
            }    

            for (auto eh : mesh.fe_range(fh)) {
                auto he0 = mesh.halfedge_handle(eh, 0);
                auto v0 = mesh.from_vertex_handle(he0);
                auto v1 = mesh.to_vertex_handle(he0);

                if (mesh.status(eh).deleted() || 
                    !mesh.data(v0).Collasable || 
                    !mesh.data(v1).Collasable) 
                    continue;

                edges.push_back(eh);
            }
        } 

        for (size_t i = 0; i < vertices.size(); ++i) {
            auto vh = vertices[i];
            mesh.data(vh).Quadric = compute_vertex_quadratic(mesh, vh);
        }

        for (size_t i = 0; i < edges.size(); ++i) {
            auto eh = edges[i];
            auto he0 = mesh.halfedge_handle(eh, 0);
            auto v0 = mesh.from_vertex_handle(he0);
            auto v1 = mesh.to_vertex_handle(he0);

            if (mesh.status(v0).deleted() || mesh.status(v1).deleted()) 
                continue;

            Eigen::Matrix4d Q = mesh.data(v0).Quadric + mesh.data(v1).Quadric;
            Eigen::Vector4d newV = compute_new_best_vertex(mesh, eh, Q);

            mesh.data(eh).Error = newV.transpose() * Q * newV;
            mesh.data(eh).NewVertex = newV;
            pq.push(eh);
        }
        deleted_faces += 2 - mesh.is_boundary(eh);
    }
}

void compute_bounding_box(const QEMMesh &mesh, Eigen::Vector3d &min, Eigen::Vector3d &max) 
{
    min = Eigen::Vector3d(std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max());
    
    max = Eigen::Vector3d(std::numeric_limits<double>::lowest(),
                          std::numeric_limits<double>::lowest(),
                          std::numeric_limits<double>::lowest());
    
    for (size_t i = 0; i < mesh.n_vertices(); ++i) {
        const auto vh = QEMMesh::VertexHandle(static_cast<int>(i));
        if (mesh.status(vh).deleted())
            continue;

        const auto coords = mesh.point(vh);
    
        if (coords[0] < min.x()) min.x() = coords[0];
        if (coords[1] < min.y()) min.y() = coords[1];
        if (coords[2] < min.z()) min.z() = coords[2];
    
        if (coords[0] > max.x()) max.x() = coords[0];
        if (coords[1] > max.y()) max.y() = coords[1];
        if (coords[2] > max.z()) max.z() = coords[2];
    }
}

Eigen::Vector4d compute_new_best_vertex(const QEMMesh& mesh, 
                                        const OpenMesh::EdgeHandle eh, 
                                        const Eigen::Matrix4d Q) 
{
    Eigen::Matrix4d quadric;
    quadric << Q(0,0), Q(0,1), Q(0,2), Q(0,3),
               Q(0,1), Q(1,1), Q(1,2), Q(1,3),
               Q(0,2), Q(1,2), Q(2,2), Q(2,3),
               0.0f,   0.0f,   0.0f,   1.0f;

    auto Error = [&](const Eigen::Vector4d& p) { return p.transpose() * Q * p; };

    if (fabs(quadric.determinant()) > 1e-12)
        return quadric.inverse() * Eigen::Vector4d(0, 0, 0, 1);
   
    else {
        auto heh = mesh.halfedge_handle(eh, 0);
        auto vh1 = mesh.from_vertex_handle(heh);
        auto vh2 = mesh.to_vertex_handle(heh);
        Eigen::Vector4d p1(mesh.point(vh1)[0], mesh.point(vh1)[1], mesh.point(vh1)[2], 1.f);
        Eigen::Vector4d p2(mesh.point(vh2)[0], mesh.point(vh2)[1], mesh.point(vh2)[2], 1.f);
        Eigen::Vector4d mid = 0.5 * (p1 + p2);

        double e1 = Error(p1);
        double e2 = Error(p2);
        double em = Error(mid);

        if (e1 <= e2 && e1 <= em)       return p1;
        else if (e2 <= e1 && e2 <= em)  return p2;
        else                            return mid;
    }
}
    
Eigen::Matrix4d compute_vertex_quadratic(const QEMMesh& mesh, 
                                         const OpenMesh::VertexHandle vh) 
{
    Eigen::Matrix4d result = Eigen::Matrix4d::Zero();
    for (auto fh : mesh.vf_range(vh)) {
        if (!mesh.status(fh).deleted())
           result += detail::compute_face_plane_mtx(mesh, fh);        
    }

    return result;
}

}
