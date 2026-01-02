#pragma once
#include <queue>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/IO/Options.hh>

#include <Eigen/Dense>
#include <cstdint>

namespace qems {

namespace detail {

struct QEMTraits : public OpenMesh::DefaultTraits {
    VertexAttributes(OpenMesh::Attributes::Color);

    VertexTraits { 
        bool Collasable = true;
        Eigen::Matrix4d Quadric; 
        std::uint32_t NodeIdx = 
            std::numeric_limits<uint32_t>::max();
    };

    EdgeTraits { 
        double Error = 0.0f;
        Eigen::Vector4d NewVertex = 
            Eigen::Vector4d::Zero();
    }; 
};
};

using QEMMesh = OpenMesh::TriMesh_ArrayKernelT<detail::QEMTraits>;

struct QEMEdgeCompare {
    const QEMMesh& mesh;

    explicit QEMEdgeCompare(const QEMMesh& m) : mesh(m) {}

    bool operator()(const QEMMesh::EdgeHandle& e1,
                    const QEMMesh::EdgeHandle& e2) const {
        return mesh.data(e1).Error > mesh.data(e2).Error;
    }
};

using QEMPriorityQueue =
    std::priority_queue<QEMMesh::EdgeHandle,
                        std::vector<QEMMesh::EdgeHandle>,
                        QEMEdgeCompare>;

}  // namespace qems
