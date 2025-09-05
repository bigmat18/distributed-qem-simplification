#include "OpenMesh/Core/Mesh/Handles.hh"
#include "utils/profiling.h"
#include <cstdint>
#include <cxxopts.hpp>
#include <iostream>
#include <iterator>
#include <mutex>
#include <ostream>
#include <queue>
#include <unistd.h>

#include <utils/utils.h>
#include <utils/mesh.h>


int main(int argc, char **argv) {
    ASSERT(argc > 1, "Need [input file]");

    cxxopts::Options options("cli", "CLI app to test distributed mesh simplification");
    options.add_options()      
        ("i,filename", "Input filename list", cxxopts::value<std::string>())
        ("n,target", "Target faces", cxxopts::value<uint32_t>());

    options.parse_positional({"filename"});
    auto result = options.parse(argc, argv);

    if(result.count("help")) {
        printf("%s", options.help().c_str()); 
        return 0;
    }

    ASSERT(result.count("filename") >= 1, "Need [input filename]");
    const std::string FILENAME        = result["filename"].as<std::string>();
    const uint32_t    TARGET_FACES    = result["target"].as<uint32_t>();

    Mesh mesh;
    ASSERT(OpenMesh::IO::read_mesh(mesh, FILENAME), "Error in mesh import");
    LOG_INFO("%s successfully imported", FILENAME.c_str());
    mesh.request_vertex_status();
    mesh.request_edge_status();
    mesh.request_face_status();
    mesh.request_halfedge_status();

    
    auto cmp = [&](const Mesh::EdgeHandle& e1, const Mesh::EdgeHandle& e2) {
        return mesh.data(e1).Error > mesh.data(e2).Error;
    };

    std::priority_queue<Mesh::EdgeHandle, 
                        std::vector<Mesh::EdgeHandle>, 
                        decltype(cmp)> pq(cmp);
    {
        PROFILING_SCOPE("CSG");

        PROFILING_LOCK();
        {
            PROFILING_SCOPE("Inizialization");

            {
                PROFILING_SCOPE("Init-Vertices-Quadratic");
                #pragma omp parallel for 
                for (int i = 0; i < mesh.n_vertices(); ++i) {
                    const auto vh = Mesh::VertexHandle(i);
                    mesh.data(vh).Quadric = EvaluateVertexQuadratic(mesh, vh);
                }
            }

            {
                PROFILING_SCOPE("Init-Edges-Quadric");
                #pragma omp parallel for
                for (int i = 0; i < mesh.n_edges(); ++i) {
                    auto eh = Mesh::EdgeHandle(i);
                    auto heh = mesh.halfedge_handle(eh, 0);
                    auto v0 = mesh.from_vertex_handle(heh);
                    auto v1 = mesh.to_vertex_handle(heh);

                    Eigen::Matrix4d Q = mesh.data(v0).Quadric + mesh.data(v1).Quadric;
                    Eigen::Vector4d newV = EvaluateNewBestVertex(mesh, eh, Q);

                    mesh.data(eh).Error = newV.transpose() * Q * newV;
                    mesh.data(eh).NewVertex = newV;
                    #pragma omp critical
                    {
                        pq.push(eh);
                    }
                } 
            }
        }
        PROFILING_UNLOCK();

        {
            PROFILING_SCOPE("Processing");
            {
                PROFILING_SCOPE("Simplification Loop");
                int deletedFaces = 0;
                while (mesh.n_faces() - deletedFaces > TARGET_FACES && !pq.empty()) {
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

                    std::vector<Mesh::VertexHandle> vertices;
                    std::vector<Mesh::EdgeHandle> edges;

                    for (auto vf_it = mesh.vf_iter(vh1); vf_it.is_valid(); ++vf_it) {
                        auto fh = *vf_it;
                        if (mesh.status(fh).deleted()) continue;

                        for (auto fv_it = mesh.fv_iter(fh); fv_it.is_valid(); ++fv_it) {
                            auto vh = *fv_it;
                            if (mesh.status(vh).deleted()) continue;
                            vertices.push_back(vh);
                        }    

                        for (auto fe_it = mesh.fe_iter(fh); fe_it.is_valid(); ++fe_it) {
                            auto eh = *fe_it;
                            if (mesh.status(eh).deleted()) continue;
                            edges.push_back(eh);
                        }
                    } 

                    #pragma omp parallel for
                    for (int i = 0; i < vertices.size(); ++i) {
                        auto vh = vertices[i];
                        mesh.data(vh).Quadric = EvaluateVertexQuadratic(mesh, vh);
                    }

                    #pragma omp parallel for
                    for (int i = 0; i < edges.size(); ++i) {
                        auto eh = edges[i];
                        auto he0 = mesh.halfedge_handle(eh, 0);
                        auto v0 = mesh.from_vertex_handle(he0);
                        auto v1 = mesh.to_vertex_handle(he0);
                        if (mesh.status(v0).deleted() || mesh.status(v1).deleted()) continue;

                        Eigen::Matrix4d Q = mesh.data(v0).Quadric + mesh.data(v1).Quadric;
                        Eigen::Vector4d newV = EvaluateNewBestVertex(mesh, eh, Q);

                        mesh.data(eh).Error = newV.transpose() * Q * newV;
                        mesh.data(eh).NewVertex = newV;
                        #pragma omp critical
                        {
                            pq.push(eh);
                        }
                    }
                    deletedFaces += 2 - mesh.is_boundary(eh);
                }
            }
            {
                PROFILING_SCOPE("Mesh Cleanup");
                mesh.garbage_collection();
            }
        }
    }

    PROFILING_PRINT();
    LOG_DEBUG("Mesh vertices: %lu, edges: %lu, faces: %lu", mesh.n_vertices(), mesh.n_edges(), mesh.n_faces());
    ASSERT(OpenMesh::IO::write_mesh(mesh, "out/out.obj"), "Error in mesh export!");
    LOG_INFO("Mesh successfully exported!");

    return 0;

} 
