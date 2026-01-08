#include "logging.hpp"
#include "massert.hpp"
#include "profiling.hpp"
#include <cstdint>
#include <cxxopts.hpp>

#include <qem_mesh.hpp>
#include <qem_simp.hpp>
#include <mesh_import.hpp>
#include <string>
#include <uniform_grid.hpp>
#include <utils.hpp>
#include <filesystem>

#include <mpi.h>

namespace fs = std::filesystem;

enum MPI_TAG {

    TAG_SIZE_V      = 0,
    TAG_DATA_V      = 1,
    TAG_SIZE_F      = 2,
    TAG_DATA_F      = 3,

    TAG_BB          = 4,
    TAG_NAME_LEN    = 5,
    TAG_NAME_DATA   = 6,
    TAG_STOP        = 7
};

int main(int argc, char* argv[]) {
    int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    cxxopts::Options options("cli", "CLI app to test distributed mesh simplification");
    options.add_options()      
        ("i,input", "Input Folder", cxxopts::value<std::string>())
        ("n,target", "Target faces", cxxopts::value<uint32_t>());

    options.parse_positional({"input"});
    auto result = options.parse(argc, argv);

    if(result.count("help")) {
        printf("%s", options.help().c_str()); 
        return 0;
    }

    const std::string INPUT           = result["input"].as<std::string>();
    const uint32_t    TARGET_FACES    = result["target"].as<uint32_t>();


    massert(fs::exists("out") && fs::is_directory("out"), 
            "out folder does not exists");

    massert(fs::exists(INPUT) && fs::is_directory(INPUT), 
            "Input must be a valid folder");


	int pid, num_procs;
	
	MPI_Comm_size(MPI_COMM_WORLD,&num_procs); 
	MPI_Comm_rank(MPI_COMM_WORLD,&pid); 

    if (pid == 0) {


        std::array<qems::MeshData, 2> buffers;
        std::array<MPI_Request, 2> requests;
        uint32_t active_idx = 0;
        bool first_iter = true;

        int worker_dest = 1;
        for(const auto& file : fs::directory_iterator(INPUT)) {
            if (!fs::is_regular_file(file.status()))
                continue;

            int load_idx = first_iter ? 0 : (1 - active_idx);
            qems::MeshData &load_data = buffers[load_idx];
            load_data.row_vertices.clear();
            load_data.row_faces.clear();

            {
                PROFILING_SCOPE("Mesh-Import");
                qems::import_mesh(file, load_data);
            }
            PROFILING_PRINT();

            if (!first_iter) {
                MPI_Waitall(2, &requests[0], MPI_STATUSES_IGNORE);
                active_idx = load_idx;
            }

            {
                PROFILING_SCOPE("Data-Sending");

                const qems::MeshData& send_data = buffers[active_idx];
                const auto& min = send_data.min_coords;
                const auto& max = send_data.max_coords;

                const uint32_t num_verts = send_data.row_vertices.size();
                const uint32_t num_faces = send_data.row_faces.size();

                double vec_buffer[6];
                vec_buffer[0] = min.x(); vec_buffer[1] = min.y(); vec_buffer[2] = min.z();
                vec_buffer[3] = max.x(); vec_buffer[4] = max.y(); vec_buffer[5] = max.z();

                MPI_Send(vec_buffer, 6, MPI_DOUBLE, worker_dest, TAG_BB, MPI_COMM_WORLD);

                uint32_t str_len = send_data.name.size();
                MPI_Send(&str_len, 1, MPI_UNSIGNED, worker_dest, TAG_NAME_LEN, MPI_COMM_WORLD);
                MPI_Send(send_data.name.c_str(), str_len, MPI_CHAR, worker_dest, TAG_NAME_DATA, MPI_COMM_WORLD);

                MPI_Send(&num_verts, 1, MPI_UNSIGNED, worker_dest, TAG_SIZE_V, MPI_COMM_WORLD);
                MPI_Isend(send_data.row_vertices.data(), num_verts, MPI_FLOAT, worker_dest, TAG_DATA_V, MPI_COMM_WORLD, &requests[0]);

                MPI_Send(&num_faces, 1, MPI_UNSIGNED, worker_dest, TAG_SIZE_F, MPI_COMM_WORLD);
                MPI_Isend(send_data.row_faces.data(), num_faces, MPI_UNSIGNED, worker_dest, TAG_DATA_F, MPI_COMM_WORLD, &requests[1]);
            }
            PROFILING_PRINT();

            worker_dest++; 
            if (worker_dest >= num_procs) worker_dest = 1;
            first_iter = false;
        }

        if (!first_iter) {
             MPI_Waitall(2, &requests[0], MPI_STATUSES_IGNORE);
        }

        double dummy_buf[6];
        for (int w = 1; w < num_procs; w++) {
            MPI_Send(dummy_buf, 6, MPI_DOUBLE, w, TAG_STOP, MPI_COMM_WORLD);
        }

    } else {
        qems::MeshData data;
        auto& min = data.min_coords;
        auto& max = data.max_coords;

        while (true) {
            data.row_faces.clear();
            data.row_vertices.clear();
            data.name.clear();

            double vec_buffer[6];
            MPI_Status status;
            MPI_Recv(&vec_buffer, 6, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_STOP)
                break;

            min.x() = vec_buffer[0]; min.y() = vec_buffer[1]; min.z() = vec_buffer[2];
            max.x() = vec_buffer[3]; max.y() = vec_buffer[4]; max.z() = vec_buffer[5];

            uint32_t str_len;
            MPI_Recv(&str_len, 1, MPI_UNSIGNED, 0, TAG_NAME_LEN, MPI_COMM_WORLD, &status);
            data.name.resize(str_len);
            MPI_Recv(&data.name[0], str_len, MPI_CHAR, 0, TAG_NAME_DATA, MPI_COMM_WORLD, &status);

            uint32_t num_verts;
            MPI_Recv(&num_verts, 1, MPI_UNSIGNED, 0, TAG_SIZE_V, MPI_COMM_WORLD, &status);

            data.row_vertices.resize(num_verts);
            MPI_Recv(data.row_vertices.data(), num_verts, MPI_FLOAT, 0, TAG_DATA_V, MPI_COMM_WORLD, &status);

            uint32_t num_faces;
            MPI_Recv(&num_faces, 1, MPI_UNSIGNED, 0, TAG_SIZE_F, MPI_COMM_WORLD, &status);

            data.row_faces.resize(num_faces);
            MPI_Recv(data.row_faces.data(), num_faces, MPI_UNSIGNED, 0, TAG_DATA_F, MPI_COMM_WORLD, &status);

            qems::QEMMesh mesh;
            qems::UniformGrid uniform_grid;

            mesh.request_vertex_status();
            mesh.request_edge_status();
            mesh.request_face_status();
            mesh.request_halfedge_status();
            LOG_INFO("{} received {} with {} vertices and {} faces", pid, data.name,
                     data.row_vertices.size() / 3, data.row_faces.size() / 3);
            {
                PROFILING_SCOPE("QEM-Simplification-Rank-" + std::to_string(pid));
                {
                    PROFILING_SCOPE("Mesh-Import");
                    qems::row_data_to_mesh(data.row_vertices, data.row_faces, mesh);
                }                


                {
                    PROFILING_SCOPE("Pre-Processing");
                    Eigen::Vector3d &min = data.min_coords;
                    Eigen::Vector3d &max = data.max_coords;

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
            PROFILING_PRINT();

            massert(OpenMesh::IO::write_mesh(mesh, "out/"+ data.name), "Error in mesh export!");
            LOG_INFO("Mesh {} successfully exported!", data.name);
        }
    }
    MPI_Finalize();
	return 0;
}

