#include <cstdint>
#include <sys/time.h>
#include <unistd.h>
#include <cxxopts.hpp>

#include <mpi.h>
#include <utils.hpp>

#include "utils.hpp"

inline void Main_Master(int pid, int num_procs,
                        const std::string INPUT,
                        const uint32_t START_PARTITIONS,
                        uint32_t NUM_MESHES,
                        const float TARGET)
{

    mpi::MessageLayout layout = get_layout();
    mpi::MPMCQueue<mpi::PackedMessage> cells_to_compute;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        if (tid == 0) {
            mpi::AsyncSend async_sender(layout, num_procs-1);

            for(int dest = 1; dest < num_procs; ++dest) {
                async_sender.wait();
                auto& msg = async_sender.get_message();
                if (!cells_to_compute.pop(msg))
                    break;
                async_sender.isend(dest);
            }

            uint32_t num_files_saved = 0;
            mpi::PackedMessage recv_msg(layout);
            while (num_files_saved < NUM_MESHES) {
                const int dest = mpi::sync_recv(recv_msg);
                const auto& recv_name = recv_msg.get_buffer<char>(CSTM_TAG_NAME);
                const auto& recv_vertices = recv_msg.get_buffer<float>(CSTM_TAG_VERT);
                const auto& recv_faces = recv_msg.get_buffer<uint32_t>(CSTM_TAG_FACE);

                if (!recv_vertices.empty()) {
                    num_files_saved++;

                    std::string str_name(recv_name.data(), recv_name.size());
                    std::vector<float> out_verts = recv_vertices;
                    std::vector<uint32_t> out_faces = recv_faces;

                    #pragma omp task firstprivate(str_name, out_verts, out_faces)
                    {
                        qems::export_mesh("out/" + str_name, recv_vertices, recv_faces);
                    }
                }

                async_sender.wait();
                auto& msg = async_sender.get_message();
                if (cells_to_compute.pop(msg)) {
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

                int counter_file = 0;
                for (const auto file : files) {
                    if (counter_file < NUM_MESHES) {
                        counter_file++;
                        #pragma omp task firstprivate(file)
                        {
                            qems::MeshMetaData metadata;
                            std::vector<float> vertices;
                            std::vector<uint32_t> faces;
                            const auto& min = metadata.min_coords;
                            const auto& max = metadata.max_coords;

                            qems::import_mesh(file, metadata, vertices, faces);

                            uint32_t cell_id = 0;
                            mpi::PackedMessage msg(layout);

                            auto& bb = msg.get_buffer<double>(CSTM_TAG_BB);
                            bb[0] = min.x(); bb[1] = min.y(); bb[2] = min.z();
                            bb[3] = max.x(); bb[4] = max.y(); bb[5] = max.z();

                            msg.get_buffer<char>(CSTM_TAG_NAME).assign(metadata.name.begin(), metadata.name.end());
                            msg.get_buffer<float>(CSTM_TAG_VERT) = std::move(vertices);
                            msg.get_buffer<uint32_t>(CSTM_TAG_FACE) = std::move(faces);

                            cells_to_compute.push(std::move(msg));
                        }
                    }
                }
            }
            cells_to_compute.signal_finished();
        }
    }
}
