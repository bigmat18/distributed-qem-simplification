#include <cstddef>
#include <cstdint>
#include <omp.h>
#include <unistd.h>
#include <cxxopts.hpp>

#include <qem_mesh.hpp>
#include <qem_simp.hpp>
#include <mesh_import.hpp>
#include <uniform_grid.hpp>
#include <utils.hpp>


int main(int argc, char **argv) {
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

    qems::MeshMetaData metadata;
    std::vector<float> vertices;
    std::vector<uint32_t> faces;

    Eigen::Vector3d &min = metadata.min_coords;
    Eigen::Vector3d &max = metadata.max_coords;

    qems::QEMMesh mesh;
    qems::UniformGrid uniform_grid;

    mesh.request_vertex_status();
    mesh.request_edge_status();
    mesh.request_face_status();
    mesh.request_halfedge_status();

    {
        PROFILING_SCOPE("QEM-Simplification");
        {
            PROFILING_SCOPE("Import-Mesh");
            qems::import_mesh(FILENAME, metadata, vertices, faces);
            qems::row_data_to_mesh(vertices, faces, mesh);
        }

        LOG_INFO("{} successfully imported", FILENAME.c_str());

        {
            PROFILING_SCOPE("Pre-Processing");

            uniform_grid = qems::UniformGrid(min, max, omp_get_max_threads() / 2);
            #pragma omp declare reduction(                                      \
                uniform_grid_merge : qems::UniformGrid : omp_out.merge(omp_in)) \
                initializer(omp_priv = qems::UniformGrid(omp_orig)              \
            )

            auto set_neighborhood = [&](qems::QEMMesh::VertexHandle vh) {
                if (!mesh.data(vh).Collasable) 
                    return;

                for (auto vvh : mesh.vv_range(vh))
                mesh.data(vvh).Collasable = false;
            };

            LOG_DEBUG("Start UniformGrid building");

            PROFILING_LOCK();
            #pragma omp parallel reduction(uniform_grid_merge : uniform_grid)
            {
                PROFILING_SCOPE("Uniform-Grid-Building");
                {
                    #pragma omp for schedule(static) 
                    for (size_t i = 0; i < mesh.n_vertices(); ++i) {
                        auto vh = qems::QEMMesh::VertexHandle(i);
                        mesh.data(vh).Quadric = qems::compute_vertex_quadratic(mesh, vh);
                        uint32_t idx = uniform_grid.add_vertex(mesh, vh);
                        mesh.data(vh).NodeIdx = idx;
                    }


                    #pragma omp for schedule(static)  
                    for (size_t i = 0; i < mesh.n_edges(); ++i) {
                        auto eh = qems::QEMMesh::EdgeHandle(i);
                        auto heh = mesh.halfedge_handle(eh, 0);
                        auto vh0 = mesh.from_vertex_handle(heh);
                        auto vh1 = mesh.to_vertex_handle(heh);

                        uint32_t idx0 = mesh.data(vh0).NodeIdx;
                        uint32_t idx1 = mesh.data(vh1).NodeIdx;

                        if (idx0 != idx1) {
                            mesh.data(vh0).Collasable = false;
                            mesh.data(vh1).Collasable = false;

                            set_neighborhood(vh0);
                            set_neighborhood(vh1);
                        } 
                    }

                    #pragma omp for schedule(static)
                    for (size_t i = 0; i < mesh.n_edges(); ++i) {
                        auto eh = qems::QEMMesh::EdgeHandle(i);

                        if(uniform_grid.add_edge(mesh, eh)) {
                            auto heh = mesh.halfedge_handle(eh, 0);
                            auto vh0 = mesh.from_vertex_handle(heh);
                            auto vh1 = mesh.to_vertex_handle(heh);

                            Eigen::Matrix4d Q = mesh.data(vh0).Quadric + mesh.data(vh1).Quadric;
                            Eigen::Vector4d newV = qems::compute_new_best_vertex(mesh, eh, Q);

                            mesh.data(eh).Error = newV.transpose() * Q * newV;
                            mesh.data(eh).NewVertex = newV;
                        }
                    }

                    #pragma omp for schedule(static)
                    for(size_t i = 0; i < mesh.n_faces(); i++) {
                        auto fh = qems::QEMMesh::FaceHandle(i);
                        uniform_grid.increment_collasable_faces(mesh, fh);
                    }
                }

            }
            PROFILING_UNLOCK();
        }

        LOG_DEBUG("Start Parallel QEM-Simplification");
        {
            PROFILING_SCOPE("Processing");

            #pragma omp parallel for schedule(dynamic, 1)
            for (const auto &cell : uniform_grid) {
                auto pq = qems::QEMPriorityQueue(qems::QEMEdgeCompare(mesh), cell.edges);

                uint32_t local_num_faces = cell.collasable_faces; 
                float total_faces = static_cast<float>(uniform_grid.total_collasable_faces());
                float cell_faces  = static_cast<float>(local_num_faces);

                float fraction = (total_faces > 0.0) ? (cell_faces / total_faces) : 0.0;
                float target_d = static_cast<float>(TARGET_FACES) * fraction;

                uint32_t local_target = static_cast<uint32_t>(std::floor(target_d));

                qems::simplification(mesh, local_target, local_num_faces, pq);
            }
        }

        {
            PROFILING_SCOPE("Mesh-Cleanup");
            mesh.garbage_collection();
        }


        LOG_DEBUG("Parallel computation mesh vertices: {}, edges: {}, faces: {}", 
                  mesh.n_vertices(), mesh.n_edges(), mesh.n_faces());
        {
            PROFILING_SCOPE("Refinements");
            std::vector<qems::QEMMesh::EdgeHandle> edges;
            edges.reserve(mesh.n_edges());

            #pragma omp parallel
            {
                std::vector<qems::QEMMesh::EdgeHandle> local_edges;
                size_t n = mesh.n_edges();
                int num_threads = omp_get_num_threads();
                local_edges.reserve((n + num_threads - 1) / num_threads);

                #pragma omp for schedule(static) 
                for (size_t i = 0; i < mesh.n_vertices(); ++i) {
                    auto vh = qems::QEMMesh::VertexHandle(i);
                    mesh.data(vh).Quadric = qems::compute_vertex_quadratic(mesh, vh);
                }
                #pragma omp for schedule(static)  
                for (size_t i = 0; i < mesh.n_edges(); ++i) {
                    auto eh = qems::QEMMesh::EdgeHandle(i);
                    auto heh = mesh.halfedge_handle(eh, 0);
                    auto vh0 = mesh.from_vertex_handle(heh);
                    auto vh1 = mesh.to_vertex_handle(heh);

                    Eigen::Matrix4d Q = mesh.data(vh0).Quadric + mesh.data(vh1).Quadric;
                    Eigen::Vector4d newV = qems::compute_new_best_vertex(mesh, eh, Q);

                    mesh.data(eh).Error = newV.transpose() * Q * newV;
                    mesh.data(eh).NewVertex = newV;
                    local_edges.push_back(eh);
                }

                #pragma omp critical
                {
                    edges.insert(edges.end(),
                                 local_edges.begin(),
                                 local_edges.end());
                }
            }

            qems::QEMPriorityQueue pq(qems::QEMEdgeCompare(mesh), edges);
            qems::simplification(mesh, TARGET_FACES, mesh.n_faces(), pq);
            mesh.garbage_collection();
        }
    }   

    LOG_DEBUG("Final computation mesh vertices: {}, edges: {}, faces: {}", 
              mesh.n_vertices(), mesh.n_edges(), mesh.n_faces());

    massert(OpenMesh::IO::write_mesh(mesh, "out/uniform_grid.ply"), "Error in mesh export!");
    LOG_INFO("Mesh successfully exported!");

    PROFILING_PRINT();
    return 0;
} 
