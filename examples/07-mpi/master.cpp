#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <cxxopts.hpp>

#include <mpi.h>
#include <utils.hpp>

#include "logging.hpp"
#include "mpmc_queue.hpp"
#include "utils.hpp"

int main (int argc, char *argv[]) {
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
        ("t,percent", "Target percent", cxxopts::value<float>()->default_value("10.0"));
 
    options.parse_positional({"input"});
    auto result = options.parse(argc, argv);
 
    const std::string INPUT             = result["input"].as<std::string>();
    const uint32_t    START_PARTITIONS  = result["partitions"].as<uint32_t>();
    const float       PERCENT           = result["percent"].as<float>();
    uint32_t          NUM_MESHES        = result["meshes"].as<uint32_t>();
    const float       TARGET            = PERCENT / 100;

    massert(fs::exists("out") && fs::is_directory("out"), 
            "out folder does not exists");

	int pid, num_procs;
	MPI_Comm_size(MPI_COMM_WORLD,&num_procs); 
	MPI_Comm_rank(MPI_COMM_WORLD,&pid); 

    {
        mpi::MessageLayout layout = get_layout();
        std::vector<mpi::MPMCQueue<mpi::PackedMessage>> cells_per_worker(num_procs-1);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            if (tid == 0) {
                mpi::AsyncSend async_sender(layout, (num_procs-1)*20);

                for (int dest = 1; dest < num_procs; ++dest) {
                    async_sender.wait();
                    auto& msg = async_sender.get_message();
                    if (!cells_per_worker[dest-1].pop(msg))
                        break;
                    async_sender.isend(dest);
                }

                uint32_t num_files_saved = 0;
                mpi::PackedMessage recv_msg(layout);

                int flag = 0;
                MPI_Status status;
                while (num_files_saved < NUM_MESHES) {

                    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
                    if (flag) {
                        mpi::sync_recv(recv_msg, status.MPI_SOURCE);
                        const auto& recv_name = recv_msg.get_element<char>(CSTM_TAG_NAME);
                        const auto& recv_vertices = recv_msg.get_buffer<float>(CSTM_TAG_VERT);
                        const auto& recv_faces = recv_msg.get_buffer<uint32_t>(CSTM_TAG_FACE);

                        if (!recv_vertices.empty()) {
                            num_files_saved++;

                            std::string str_name(recv_name.data(), recv_name.size());
                            std::vector<float> out_verts = recv_vertices;
                            std::vector<uint32_t> out_faces = recv_faces;

                            #pragma omp task firstprivate(str_name, out_verts, out_faces)
                            {
                                qems::export_mesh("out/" + str_name, out_verts, out_faces);
                            }
                        }
                    }

                    int free_buffer_id = async_sender.wait();
                    free_buffer_id = free_buffer_id % (num_procs-1);
                    auto& msg = async_sender.get_message();
                    if (cells_per_worker[free_buffer_id].pop(msg)) {
                        async_sender.isend(free_buffer_id+1);
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
                    std::vector<fs::path> files;
                    if (!fs::is_directory(INPUT)) {
                        files.push_back(INPUT);
                        NUM_MESHES = 1;
                    } else {
                        for (const auto file : fs::directory_iterator(INPUT)) {
                            if (!fs::is_regular_file(file.status()))
                                continue;
                            files.push_back(file); 
                        }
                    }

                    uint32_t counter_file = 0;
                    for (const auto file : files) {
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
                                    metadata.max_coords, START_PARTITIONS 
                                );

                                uint32_t cell_id = 0;
                                float total_faces = static_cast<float>(faces.size()/3);
                                uint32_t final_target = static_cast<uint32_t>(std::floor(total_faces * TARGET));
                                #pragma omp critical(file_ordering)
                                {
                                    for (auto &cell : uniform_grid) {
                                        mpi::PackedMessage msg(layout);

                                        msg.get_element<uint32_t>(CSTM_TAG_CELL_PART_LVL) = {START_PARTITIONS, START_PARTITIONS};
                                        msg.get_element<uint32_t>(CSTM_TAG_FINAL_TARGET) = { final_target };

                                        auto& bb = msg.get_element<double>(CSTM_TAG_BB);
                                        bb[0] = min.x(); bb[1] = min.y(); bb[2] = min.z();
                                        bb[3] = max.x(); bb[4] = max.y(); bb[5] = max.z();

                                        msg.get_element<char>(CSTM_TAG_NAME).assign(metadata.name.begin(), metadata.name.end());
                                        msg.get_element<uint32_t>(CSTM_TAG_CELL_ID) = {cell_id, cell_id, counter_file};

                                        msg.get_buffer<float>(CSTM_TAG_VERT) = std::move(cell.vertices);
                                        msg.get_buffer<uint32_t>(CSTM_TAG_FACE) = std::move(cell.faces);
                                        msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP) = std::move(cell.indices_mapping);

                                        uint32_t dest = get_dest(cell_id, START_PARTITIONS, num_procs-1, counter_file);
                                        cells_per_worker[dest].push(std::move(msg));
                                        cell_id++;
                                    }
                                }
                            }
                            counter_file++;
                        }
                    }
                }
                for (int w = 0; w < num_procs-1; w++)
                    cells_per_worker[w].signal_finished();
            }
        }
    }

    MPI_Finalize();
    return 0;
}
