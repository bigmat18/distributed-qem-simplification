#pragma once
#include "qem_mesh.hpp"

namespace qems {

namespace detail {

inline Eigen::Vector4d compute_face_plane(const QEMMesh& mesh, 
                                          const OpenMesh::FaceHandle fh) 
{

    std::array<Eigen::Vector3d, 3> points;
    int i = 0;
    for (const auto& vh : mesh.fv_range(fh)) {
        auto p = mesh.point(vh);
        points[i++] = Eigen::Vector3d(p[0], p[1], p[2]);
    }

    Eigen::Vector3d u = points[1] - points[0];
    Eigen::Vector3d v = points[2] - points[0];
    Eigen::Vector3d n = u.cross(v);
    n.normalize();

    double a = n.x();
    double b = n.y();
    double c = n.z();
    double d = -n.dot(points[0]);

    return Eigen::Vector4d(a, b, c, d);
}

inline Eigen::Matrix4d compute_face_plane_mtx(const QEMMesh& mesh, 
                                              const OpenMesh::FaceHandle fh)
{
    Eigen::Vector4d planeCoeficient = compute_face_plane(mesh, fh);
    return planeCoeficient * planeCoeficient.transpose();
}

}


void simplification(QEMMesh &mesh, uint32_t target, uint32_t num_faces, QEMPriorityQueue &pq);

void compute_bounding_box(const QEMMesh &mesh, Eigen::Vector3d &min, Eigen::Vector3d &max);

Eigen::Vector4d compute_new_best_vertex(const QEMMesh& mesh, 
                                        const OpenMesh::EdgeHandle eh, 
                                        const Eigen::Matrix4d Q);
    
Eigen::Matrix4d compute_vertex_quadratic(const QEMMesh& mesh, 
                                         const OpenMesh::VertexHandle vh);
}
