#include <algorithm>
#include <cstdint>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <cxxopts.hpp>

#include <mpi.h>
#include <utils.hpp>

#include "logging.hpp"
#include "mesh_import.hpp"
#include "packed_message.hpp"
#include "ug_row_data.hpp"
#include "utils.hpp"

int main(int argc, char *argv[]) {
    omlog().disable();
    omout().disable();
    omerr().disable();

    int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    cxxopts::Options options("cli", "CLI app to test distributed mesh simplification");
    options.add_options()      
        ("i,input", "Input Folder", cxxopts::value<std::string>())
        ("n,meshes", "Num meshes", cxxopts::value<uint32_t>())
        ("p,partitions", "Start partitions", cxxopts::value<uint32_t>()->default_value("4"))
        ("t,percent", "Target percent", cxxopts::value<uint32_t>()->default_value("10"));
 
    options.parse_positional({"input"});
    auto result = options.parse(argc, argv);
 

    const std::string INPUT             = result["input"].as<std::string>();
    const uint32_t    NUM_MESHES        = result["meshes"].as<uint32_t>();
    const uint32_t    PERCENT           = result["percent"].as<uint32_t>();
    const float       TARGET            = static_cast<float>(PERCENT) / 100;
    const uint32_t    GLOBAL_PARTITIONS = 2;

    massert(fs::exists("out") && fs::is_directory("out"), 
            "out folder does not exists");
    massert(fs::exists(INPUT) && fs::is_directory(INPUT), 
            "Input must be a valid folder");

	int pid, num_procs;
	MPI_Comm_size(MPI_COMM_WORLD,&num_procs); 
	MPI_Comm_rank(MPI_COMM_WORLD,&pid); 

    {
        mpi::MessageLayout layout = get_layout();
        mpi::MPMCQueue<mpi::PackedMessage> cells_per_worker;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            if (tid == 0) {
                mpi::AsyncSend async_sender(layout, (num_procs-1)*50);
                std::unordered_map<std::string, std::vector<mpi::PackedMessage>> mapping;

                for (int dest = 1; dest < num_procs; ++dest) {
                    async_sender.wait();
                    auto& msg = async_sender.get_message();
                    if (!cells_per_worker.pop(msg))
                        break;
                    async_sender.isend(dest);
                }

                uint32_t num_files_saved = 0;
                mpi::PackedMessage recv_msg(layout);
                while (num_files_saved < NUM_MESHES) {
                    const int dest = mpi::sync_recv(recv_msg);
                    const auto& recv_name = recv_msg.get_element<char>(CSTM_TAG_NAME);
                    std::string str_name(recv_name.data(), recv_name.size());
                    mapping[str_name].push_back(recv_msg);

                    if (mapping[str_name].size() == 8) {
                        num_files_saved++;
                        std::vector<mpi::PackedMessage> tmp_vec = std::move(mapping[str_name]);

                        #pragma omp task firstprivate(tmp_vec, str_name)
                        {
                            uint32_t final_target = tmp_vec[0].get_element<uint32_t>(CSTM_TAG_FINAL_TARGET)[0];
                            auto& bb = tmp_vec[0].get_element<double>(CSTM_TAG_BB);
                            Eigen::Vector3d min, max;
                            min.x() = bb[0]; min.y() = bb[1]; min.z() = bb[2];
                            max.x() = bb[3]; max.y() = bb[4]; max.z() = bb[5];

                            mpi::UniformGridRow uniform_grid({}, {}, {}, min, max, GLOBAL_PARTITIONS);  
                            for(auto& msg : tmp_vec) {
                                uint32_t id = msg.get_element<uint32_t>(CSTM_TAG_CELL_ID)[0];
                                auto& cell = uniform_grid.cells()[id];
                                cell.vertices = std::move(msg.get_buffer<float>(CSTM_TAG_VERT));
                                cell.faces = std::move(msg.get_buffer<uint32_t>(CSTM_TAG_FACE));
                                cell.indices_mapping = std::move(msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP));
                            }
                            std::vector<float> out_vertices;
                            std::vector<uint32_t> out_faces;
                            std::vector<uint32_t> out_idx_map;
                            uniform_grid.merge_cells(out_vertices, out_faces, out_idx_map);


                            qems::QEMMesh mesh;
                            mesh.request_vertex_status();
                            mesh.request_edge_status();
                            mesh.request_face_status();
                            mesh.request_halfedge_status();
                            mesh.request_vertex_normals();
                            mesh.request_face_normals();

                            qems::row_data_to_mesh(out_vertices, out_faces, out_idx_map, mesh);
                            {
                                mesh.update_normals();
                                std::vector<qems::QEMMesh::EdgeHandle> edges;

                                for (auto vh : mesh.vertices()) {
                                    auto coords = mesh.point(vh);
                                    mesh.data(vh).Quadric = qems::compute_vertex_quadratic(mesh, vh);
                                }


                                for (auto eh : mesh.edges()) {
                                    auto heh = mesh.halfedge_handle(eh, 0);
                                    auto vh0 = mesh.from_vertex_handle(heh);
                                    auto vh1 = mesh.to_vertex_handle(heh);

                                    Eigen::Matrix4d Q = mesh.data(vh0).Quadric + mesh.data(vh1).Quadric;
                                    Eigen::Vector4d newV = qems::compute_new_best_vertex(mesh, eh, Q);

                                    mesh.data(eh).Error = newV.transpose() * Q * newV;
                                    mesh.data(eh).NewVertex = newV;
                                    edges.push_back(eh);
                                }

                                auto pq = qems::QEMPriorityQueue(qems::QEMEdgeCompare(mesh), edges);

                                qems::simplification(mesh,final_target, mesh.n_faces(), pq);
                                mesh.garbage_collection();
                            }
                            qems::mesh_to_row_data(mesh, out_vertices, out_faces, out_idx_map);
                            qems::export_mesh("out/"+str_name, out_vertices, out_faces);
                        }
                    }

                    async_sender.wait();
                    auto& msg = async_sender.get_message();
                    if (cells_per_worker.pop(msg)) {
                        async_sender.isend(dest);
                    }
                }

                mpi::PackedMessage final_msg;
                for (int w = 1; w < num_procs; ++w)
                    mpi::sync_send(w, final_msg);
            }

            #pragma omp single nowait 
            {
                #pragma omp taskgroup 
                {
                    uint32_t counter_file = 0;
                    for (const auto file : fs::directory_iterator(INPUT)) {
                        if (!fs::is_regular_file(file.status()))
                            continue;

                        if (counter_file < NUM_MESHES) {
                            #pragma omp task firstprivate(file, counter_file)
                            {
                                qems::MeshMetaData metadata;
                                std::vector<float> vertices;
                                std::vector<uint32_t> faces;
                                const auto& min = metadata.min_coords;
                                const auto& max = metadata.max_coords;

                                qems::import_mesh(file, metadata, vertices, faces);
                                auto uniform_grid = mpi::UniformGridRow(
                                    vertices, faces, metadata.min_coords, 
                                    metadata.max_coords, GLOBAL_PARTITIONS 
                                );

                                uint32_t cell_id = 0;
                                uint32_t final_target = static_cast<uint32_t>(static_cast<float>(faces.size()/3) * TARGET);
                                #pragma omp critical(file_ordering)
                                {
                                    for (auto &cell : uniform_grid) {
                                        mpi::PackedMessage msg(layout);

                                        auto& bb = msg.get_element<double>(CSTM_TAG_BB);
                                        bb[0] = min.x(); bb[1] = min.y(); bb[2] = min.z();
                                        bb[3] = max.x(); bb[4] = max.y(); bb[5] = max.z();

                                        msg.get_element<char>(CSTM_TAG_NAME).assign(metadata.name.begin(), metadata.name.end());
                                        msg.get_element<uint32_t>(CSTM_TAG_CELL_ID) = {cell_id};
                                        msg.get_element<uint32_t>(CSTM_TAG_FINAL_TARGET) = {final_target};
                                        msg.get_buffer<float>(CSTM_TAG_VERT) = std::move(cell.vertices);
                                        msg.get_buffer<uint32_t>(CSTM_TAG_FACE) = std::move(cell.faces);
                                        msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP) = std::move(cell.indices_mapping);

                                        cells_per_worker.push(std::move(msg));
                                        cell_id++;
                                    }
                                }
                            }
                            counter_file++;
                        }
                    }
                }
                cells_per_worker.signal_finished();
            }
        }
    }

    MPI_Finalize();
    return 0;
}
