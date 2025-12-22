#ifndef MESH_H
#define MESH_H

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <Eigen/Dense>

struct Traits : public OpenMesh::DefaultTraits {
    VertexTraits { 
        Eigen::Matrix4d Quadric; 
    };

    EdgeTraits { 
        double Error;
        Eigen::Vector4d NewVertex;
    };
};

using Mesh = OpenMesh::TriMesh_ArrayKernelT<Traits>;

inline bool CompareMeshEdge(const Mesh& mesh, const Mesh::EdgeHandle& e1, const Mesh::EdgeHandle& e2) {
    return mesh.data(e1).Error > mesh.data(e2).Error;
};

inline Eigen::Vector4d EvaluateFacePlane(Mesh& mesh, 
                                         const OpenMesh::FaceHandle fh) 
{

    std::array<Eigen::Vector3f, 3> points;
    int i = 0;
    for (auto fv_it = mesh.fv_iter(fh); fv_it.is_valid(); ++fv_it) {
        auto p = mesh.point(*fv_it);
        points[i++] = Eigen::Vector3f(p[0], p[1], p[2]);
    }

    Eigen::Vector3f u = points[1] - points[0];
    Eigen::Vector3f v = points[2] - points[0];
    Eigen::Vector3f n = u.cross(v);
    n.normalize();

    float a = n.x();
    float b = n.y();
    float c = n.z();
    float d = -n.dot(points[0]);

    return Eigen::Vector4d(a, b, c, d);
}

inline Eigen::Matrix4d EvaluateFacePlaneMatrix(Mesh& mesh, 
                                               const OpenMesh::FaceHandle fh)
{
    Eigen::Vector4d planeCoeficient = EvaluateFacePlane(mesh, fh);
    return planeCoeficient * planeCoeficient.transpose();
}

inline Eigen::Matrix4d EvaluateVertexQuadratic(Mesh& mesh, 
                                               const OpenMesh::VertexHandle vh)
{
    Eigen::Matrix4d result = Eigen::Matrix4d::Zero();
    for (auto f_it = mesh.vf_iter(vh); f_it.is_valid(); ++f_it) {
        auto fh = *f_it; 
        result += EvaluateFacePlaneMatrix(mesh, fh);        
    }

    return result;
}

inline Eigen::Vector4d EvaluateNewBestVertex(const Mesh& mesh, 
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

#endif
