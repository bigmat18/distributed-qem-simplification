#pragma once
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

using QEMMesh = OpenMesh::TriMesh_ArrayKernelT<Traits>;


