#pragma once
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/IO/Options.hh>
#include <Eigen/Dense>

struct Traits : public OpenMesh::DefaultTraits {
    VertexAttributes(OpenMesh::Attributes::Color);

    VertexTraits { 
        bool Collapable = true;
        Eigen::Matrix4d Quadric; 
    };

    EdgeTraits { 
        double Error;
        Eigen::Vector4d NewVertex;
    }; 
};

using QEMMesh = OpenMesh::TriMesh_ArrayKernelT<Traits>;


