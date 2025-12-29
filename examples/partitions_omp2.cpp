#include "logging.hpp"
#include "profiling.hpp"
#include <cstddef>
#include <cstdint>
#include <cxxopts.hpp>
#include <unistd.h>
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
            PROFILING_SCOPE("QEM");

            #pragma omp for schedule(static)
            for (size_t j = 0; j < num_cells; j++) { 
                uint32_t local_num_faces = uniform_grid.collasable_faces(j);
                float total_faces = static_cast<float>(uniform_grid.total_collasable_faces());
                float cell_faces  = static_cast<float>(local_num_faces);
                
                float fraction = (total_faces > 0.0) ? (cell_faces / total_faces) : 0.0;
                float target_d = static_cast<float>(TARGET_FACES) * fraction;
                
                uint32_t local_target = static_cast<uint32_t>(std::floor(target_d));

                pqs[j] = uniform_grid.get_qem_pq(mesh, j);
                ComputeQEMSimplification(mesh, local_target, local_num_faces, pqs[j]);
            }
        }
        PROFILING_UNLOCK();

        {
            PROFILING_SCOPE("Mesh Cleanup");
            mesh.garbage_collection();
        }

        {
            PROFILING_SCOPE("QEM Refinements");
            std::vector<QEMMesh::EdgeHandle> elements;
            {
                PROFILING_SCOPE("Init Vertices Quadratic");
                #pragma omp for schedule(static) 
                for (size_t i = 0; i < mesh.n_vertices(); ++i) {
                    auto vh = QEMMesh::VertexHandle(i);
                    mesh.data(vh).Quadric = EvaluateVertexQuadratic(mesh, vh);
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

                    Eigen::Matrix4d Q = mesh.data(vh0).Quadric + mesh.data(vh1).Quadric;
                    Eigen::Vector4d newV = EvaluateNewBestVertex(mesh, eh, Q);

                    mesh.data(eh).Error = newV.transpose() * Q * newV;
                    mesh.data(eh).NewVertex = newV;
                    elements.push_back(eh);
                }
            }
            {
                PROFILING_SCOPE("QEM Computation");
                QEMPriorityQueue pq(QEMEdgeCompare(&mesh), std::move(elements));
                ComputeQEMSimplification(mesh, TARGET_FACES, mesh.n_faces(), pq);
            }
        }

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
