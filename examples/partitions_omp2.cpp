#include "logging.hpp"
#include "profiling.hpp"
#include <cstddef>
#include <cstdint>
#include <cxxopts.hpp>
#include <unistd.h>
#include <queue>
#include <utils.hpp>

#include <qem_mesh.hpp>
#include <qem_utils.hpp>
#include <uniform_grid.hpp>
#include <format_utils.hpp>

using Mesh = QEMMesh;
#define QUAD_PARTITIONS 4

int main(int argc, char **argv) {
    massert(argc > 1, "Need [input file]");

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

    massert(result.count("filename") >= 1, "Need [input filename]");
    const std::string FILENAME        = result["filename"].as<std::string>();
    const uint32_t    TARGET_FACES    = result["target"].as<uint32_t>();

    Mesh mesh;
    massert(OpenMesh::IO::read_mesh(mesh, FILENAME), "Error in mesh import");
    LOG_INFO("{} successfully imported", FILENAME.c_str());
    mesh.request_vertex_status();
    mesh.request_vertex_colors();
    mesh.request_edge_status();
    mesh.request_face_status();
    mesh.request_halfedge_status();
    

    {
        PROFILING_SCOPE("Computation");
        Eigen::Vector3f min;
        Eigen::Vector3f max;

        ComputeBoundingBox(mesh, min, max); 
        LOG_INFO("Bounding Box: {}, {}", min, max);

        #pragma omp declare reduction(                                  \
            uniform_grid_merge : UniformGrid<QUAD_PARTITIONS> : omp_out.merge(omp_in))\
            initializer(omp_priv = UniformGrid<QUAD_PARTITIONS>(omp_orig))

        UniformGrid<QUAD_PARTITIONS> uniform_grid(min, max);

        PROFILING_LOCK();
        #pragma omp parallel reduction(uniform_grid_merge : uniform_grid)
        {
            PROFILING_SCOPE("UniformGrid Building");

            {
                PROFILING_SCOPE("Init Vertices Quadratic");
                #pragma omp for schedule(static) 
                for (size_t i = 0; i < mesh.n_vertices(); ++i) {
                    auto vh = QEMMesh::VertexHandle(i);
                    mesh.data(vh).Quadric = EvaluateVertexQuadratic(mesh, vh);
                    uniform_grid.add_vertex(mesh, vh);
                }
            }

            {
                PROFILING_SCOPE("Init Edges Quadratic");
                #pragma omp for schedule(static)  
                for (size_t i = 0; i < mesh.n_edges(); ++i) {
                    auto eh = QEMMesh::EdgeHandle(i);
                    auto heh = mesh.halfedge_handle(eh, 0);
                    auto vh0 = mesh.from_vertex_handle(heh);
                    auto vh1 = mesh.to_vertex_handle(heh);

                    size_t idx0 = uniform_grid.get_vertex_indices(mesh, vh0).w();
                    size_t idx1 = uniform_grid.get_vertex_indices(mesh, vh1).w();

                    if (idx0 == idx1) {
                        Eigen::Matrix4d Q = mesh.data(vh0).Quadric + mesh.data(vh1).Quadric;
                        Eigen::Vector4d newV = EvaluateNewBestVertex(mesh, eh, Q);

                        mesh.data(eh).Error = newV.transpose() * Q * newV;
                        mesh.data(eh).NewVertex = newV;
                    } else {
                        mesh.data(vh0).Collapable = false;
                        mesh.data(vh0).Collapable = false;
                    }
                }
            }

            {
                PROFILING_SCOPE("Count Faces per Quad");
                #pragma omp for schedule(static)
                for(size_t i = 0; i < mesh.n_faces(); i++) {
                    auto fh = QEMMesh::FaceHandle(i);
                    uniform_grid.increment_collasable_faces(mesh, fh);
                }
            }

            {
                PROFILING_SCOPE("Set Collapsable Edges");
                #pragma omp for schedule(static)
                for (size_t i = 0; i < mesh.n_edges(); ++i) {
                    auto eh = QEMMesh::EdgeHandle(i);
                    uniform_grid.add_edge(mesh, eh);
                }
            }
        }
        PROFILING_UNLOCK();

        constexpr size_t num_cells = QUAD_PARTITIONS * QUAD_PARTITIONS * QUAD_PARTITIONS;
        std::vector<QEMPriorityQueue> pqs(num_cells);

        PROFILING_LOCK();
        #pragma omp parallel
        {
            PROFILING_SCOPE("QEM Computation");

            #pragma omp for schedule(static)
            for (size_t j = 0; j < num_cells; j++) { 
                uint32_t local_num_faces = uniform_grid.collasable_faces(j);
                float total_faces = static_cast<float>(mesh.n_faces());
                float cell_faces  = static_cast<float>(local_num_faces);
                
                float fraction = (total_faces > 0.0) ? (cell_faces / total_faces) : 0.0;
                float target_d = static_cast<float>(TARGET_FACES) * fraction;
                
                uint32_t local_target = static_cast<uint32_t>(std::floor(target_d));

                pqs[j] = uniform_grid.get_qem_pq(mesh, j);

                uint32_t deleted_faces = 0;
                while (local_num_faces - deleted_faces > local_target && !pqs[j].empty()) { 
                    auto eh = pqs[j].top();
                    pqs[j].pop();

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
                            if (mesh.status(vh).deleted() || !mesh.data(vh).Collapable) 
                                continue;

                            vertices.push_back(vh);
                        }    

                        for (auto fe_it = mesh.fe_iter(fh); fe_it.is_valid(); ++fe_it) {
                            auto eh = *fe_it;

                            auto he0 = mesh.halfedge_handle(eh, 0);
                            auto v0 = mesh.from_vertex_handle(he0);
                            auto v1 = mesh.to_vertex_handle(he0);

                            if (mesh.status(eh).deleted() || 
                                !mesh.data(v0).Collapable || 
                                !mesh.data(v1).Collapable) 
                                continue;

                            edges.push_back(eh);
                        }
                    } 

                    for (size_t i = 0; i < vertices.size(); ++i) {
                        auto vh = vertices[i];
                        mesh.data(vh).Quadric = EvaluateVertexQuadratic(mesh, vh);
                    }

                    for (size_t i = 0; i < edges.size(); ++i) {
                        auto eh = edges[i];
                        auto he0 = mesh.halfedge_handle(eh, 0);
                        auto v0 = mesh.from_vertex_handle(he0);
                        auto v1 = mesh.to_vertex_handle(he0);
                        if (mesh.status(v0).deleted() || mesh.status(v1).deleted()) continue;

                        Eigen::Matrix4d Q = mesh.data(v0).Quadric + mesh.data(v1).Quadric;
                        Eigen::Vector4d newV = EvaluateNewBestVertex(mesh, eh, Q);

                        mesh.data(eh).Error = newV.transpose() * Q * newV;
                        mesh.data(eh).NewVertex = newV;
                        pqs[j].push(eh);
                    }
                    deleted_faces += 2 - mesh.is_boundary(eh);
                }
            }
        }
        PROFILING_UNLOCK();
        {
            PROFILING_SCOPE("Mesh Cleanup");
            mesh.garbage_collection();
        }
    }   

    PROFILING_PRINT();

    OpenMesh::IO::Options wopt;
    wopt += OpenMesh::IO::Options::VertexColor;

    massert(OpenMesh::IO::write_mesh(mesh, "out/out.ply", wopt), "Error in mesh export!");
    LOG_INFO("Mesh successfully exported!");
    return 0;
} 
