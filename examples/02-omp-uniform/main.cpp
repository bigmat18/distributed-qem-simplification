#include <cstddef>
#include <cstdint>
#include <omp.h>
#include <string>
#include <unistd.h>
#include <cxxopts.hpp>

#include <qem_mesh.hpp>
#include <qem_simp.hpp>
#include <mesh_import.hpp>
#include <uniform_grid_qem.hpp>
#include <utils.hpp>

inline int next_step(int n) {
    if (n == 1) return 0;
    return (n % 2 != 0) ? (n - 1) / 2 : (n > 4 ? n / 2 + 1 : 1);
}


int main(int argc, char **argv) {
    omlog().disable();
    omout().disable();
    omerr().disable();

    cxxopts::Options options("cli", "CLI app to test distributed mesh simplification");
    options.add_options()      
        ("i,filename", "Input filename list", cxxopts::value<std::string>())
        ("t,threads", "Num threads", cxxopts::value<int>()->default_value("-1"))
        ("p,partitions", "Start partitions", cxxopts::value<uint32_t>()->default_value("16"))
        ("n,target", "Target faces", cxxopts::value<uint32_t>());

    options.parse_positional({"filename"});
    auto result = options.parse(argc, argv);

    if(result.count("help")) {
        printf("%s", options.help().c_str()); 
        return 0;
    }

    massert(result.count("filename") >= 1, "Need [input filename]");
    const int         NUM_THREAD        = result["threads"].as<int>();
    const std::string FILENAME          = result["filename"].as<std::string>();
    const uint32_t    TARGET_FACES      = result["target"].as<uint32_t>();
    const uint32_t    START_PARTITIONS  = result["partitions"].as<uint32_t>();

    if (NUM_THREAD != -1)
        omp_set_num_threads(NUM_THREAD);

    qems::MeshMetaData metadata;
    std::vector<float> vertices;
    std::vector<uint32_t> faces;

    Eigen::Vector3d &min = metadata.min_coords;
    Eigen::Vector3d &max = metadata.max_coords;

    qems::QEMMesh mesh;
    qems::UniformGridQEM uniform_grid;

    mesh.request_vertex_status();
    mesh.request_vertex_colors();
    mesh.request_edge_status();
    mesh.request_face_status();
    mesh.request_halfedge_status();
    mesh.request_vertex_normals();
    mesh.request_face_normals();

    {
        PROFILING_SCOPE("QEM-Simplification");
        {
            PROFILING_SCOPE("Import-mesh");
            qems::import_mesh(FILENAME, metadata, vertices, faces);
            qems::row_data_to_mesh(vertices, faces, mesh);
            mesh.update_normals();
        }

        LOG_DEBUG("{} successfully imported", FILENAME.c_str());
        uint32_t subdivision = START_PARTITIONS;

        while (subdivision > 0 && mesh.n_faces() > TARGET_FACES) {
            PROFILING_SCOPE("Iteration-" + std::to_string(subdivision));
            {
                PROFILING_SCOPE("Pre-Processing");

                uniform_grid = qems::UniformGridQEM(min, max, subdivision);
                #pragma omp declare reduction(                                      \
                    uniform_grid_merge : qems::UniformGridQEM : omp_out.merge(omp_in)) \
                    initializer(omp_priv = qems::UniformGridQEM(omp_orig)              \
                )

                LOG_DEBUG("Start UniformGridQEM building");

                #pragma omp parallel reduction(uniform_grid_merge : uniform_grid)
                {
                    #pragma omp for schedule(static) 
                    for (size_t i = 0; i < mesh.n_vertices(); ++i) {
                        auto vh = qems::QEMMesh::VertexHandle(i);
                        mesh.data(vh).Quadric = qems::compute_vertex_quadratic(mesh, vh);
                        uint32_t idx = uniform_grid.add_vertex(mesh, vh);
                        mesh.data(vh).NodeIdx = idx;
                        mesh.data(vh).Collasable = true;
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

            OpenMesh::IO::Options opt;
            opt += OpenMesh::IO::Options::VertexColor;
            massert(OpenMesh::IO::write_mesh(mesh, "out/" + std::to_string(subdivision) + "_" + metadata.name, opt), "Error in mesh export!");

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

            subdivision = next_step(subdivision);
            LOG_DEBUG("Parallel computation mesh vertices: {}, edges: {}, faces: {}", 
                      mesh.n_vertices(), mesh.n_edges(), mesh.n_faces());
        }

        {
            PROFILING_SCOPE("Export-Mesh");
            massert(OpenMesh::IO::write_mesh(mesh, "out/" + metadata.name), "Error in mesh export!");
            LOG_DEBUG("Mesh successfully exported!");
        }

    }

    PROFILING_PRINT();
    return 0;
} 
