#include <cstdint>
#include <sys/time.h>
#include <unistd.h>
#include <cxxopts.hpp>

#include <mpi.h>
#include <utils.hpp>

#include "profiling.hpp"
#include "utils.hpp"

int main(int argc, char **argv) {
    omlog().disable();
    omout().disable();
    omerr().disable();

    int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    cxxopts::Options options("cli", "CLI app to test distributed mesh simplification");
    options.add_options()      
        ("p,partitions", "Start partitions", cxxopts::value<uint32_t>()->default_value("4"))
        ("t,percent", "Target percent", cxxopts::value<uint32_t>()->default_value("10"));
 
    options.parse_positional({"input"});
    auto result = options.parse(argc, argv);
 
    const uint32_t    START_PARTITIONS  = result["partitions"].as<uint32_t>();
    const uint32_t    PERCENT           = result["percent"].as<uint32_t>();
    const float       TARGET            = static_cast<float>(PERCENT) / 100;

	int pid, num_procs;
	MPI_Comm_size(MPI_COMM_WORLD,&num_procs); 
	MPI_Comm_rank(MPI_COMM_WORLD,&pid); 

    {
        mpi::MessageLayout layout = get_layout();
        mpi::PackedMessage msg(layout);
        auto& name = msg.get_buffer<char>(CSTM_TAG_NAME);
        auto& bb = msg.get_buffer<double>(CSTM_TAG_BB);
        auto& vertices = msg.get_buffer<float>(CSTM_TAG_VERT);
        auto& faces = msg.get_buffer<uint32_t>(CSTM_TAG_FACE);

        Eigen::Vector3d min, max;
        while (true) {
            if (mpi::sync_recv(msg, 0) == -1) 
                break;

            std::string str_name(name.data(), name.size());

            min.x() = bb[0]; min.y() = bb[1]; min.z() = bb[2]; 
            max.x() = bb[3]; max.y() = bb[4]; max.z() = bb[5];

            qems::QEMMesh mesh;
            qems::UniformGrid uniform_grid;

            mesh.request_vertex_status();
            mesh.request_edge_status();
            mesh.request_face_status();
            mesh.request_halfedge_status();
            mesh.request_vertex_normals();
            mesh.request_face_normals();

            {
                PROFILING_SCOPE("PID:"+ std::to_string(pid) +",Mesh:" + str_name);
                qems::row_data_to_mesh(vertices, faces, mesh);
                mesh.update_normals();

                const uint32_t TARGET_FACES = static_cast<uint32_t>(mesh.n_faces() * TARGET);
                LOG_DEBUG("{} - Received {} with {} vertices, {} faces, target {}", 
                          pid, str_name,
                          vertices.size() / 3, 
                          faces.size() / 3,
                          TARGET_FACES);


                uint32_t subdivision = START_PARTITIONS;
                while (subdivision > 0 && mesh.n_faces() > TARGET_FACES) {
                    uniform_grid = qems::UniformGrid(min, max, subdivision);

                    #pragma omp declare reduction(                                      \
                        uniform_grid_merge : qems::UniformGrid : omp_out.merge(omp_in)) \
                        initializer(omp_priv = qems::UniformGrid(omp_orig))


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

                    mesh.garbage_collection();
                    subdivision = next_step(subdivision);
                }
            }
            PROFILING_PRINT();

            qems::mesh_to_row_data(mesh, vertices, faces);
            mpi::sync_send(0, msg);
        }
    }

    MPI_Finalize();
    return 0;
}
