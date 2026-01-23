#include <cstdint>
#include <sys/time.h>
#include <unistd.h>
#include <cxxopts.hpp>

#include <mpi.h>
#include <utils.hpp>

#include "utils.hpp"

int main(int argc, char **argv) {
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
    const uint32_t    START_PARTITIONS  = result["partitions"].as<uint32_t>();
    const uint32_t    PERCENT           = result["percent"].as<uint32_t>();
    const float       TARGET            = static_cast<float>(PERCENT) / 100;

    massert(fs::exists("out") && fs::is_directory("out"), 
            "out folder does not exists");
    massert(fs::exists(INPUT) && fs::is_directory(INPUT), 
            "Input must be a valid folder");

	int pid, num_procs;
	MPI_Comm_size(MPI_COMM_WORLD,&num_procs); 
	MPI_Comm_rank(MPI_COMM_WORLD,&pid); 

    {
        mpi::MessageLayout layout = get_layout();

        std::vector<fs::path> files;
        uint32_t file_insert = 0;
        for (const auto& file : fs::directory_iterator(INPUT)) {
            if (file_insert >= NUM_MESHES)
                break;

            if (fs::is_regular_file(file.status())) {
                files.push_back(file);
                file_insert++;
            }
        }

        uint32_t current_file_idx = 0;
        uint32_t active_workers = 0;

        qems::MeshMetaData metadata;
        mpi::AsyncSend send(layout);

        const auto& min = metadata.min_coords;
        const auto& max = metadata.max_coords;

        for (int dest = 1; dest < num_procs; ++dest) {
            if (current_file_idx >= files.size()) 
                break;  

            const auto& file = files[current_file_idx];

            {
                PROFILING_SCOPE("Sending-" + file.filename().string());

                send.wait();

                auto& msg = send.get_message();

                auto& name = msg.get_buffer<char>(CSTM_TAG_NAME);
                auto& bb = msg.get_buffer<double>(CSTM_TAG_BB);
                auto& vertices = msg.get_buffer<float>(CSTM_TAG_VERT);
                auto& faces = msg.get_buffer<uint32_t>(CSTM_TAG_FACE);

                qems::import_mesh(file, metadata, vertices, faces);

                LOG_DEBUG("{} - Imported {} with {} vertices, {} faces", 
                          pid, metadata.name,
                          vertices.size() / 3, 
                          faces.size() / 3);

                name.clear();
                name.assign(metadata.name.begin(), metadata.name.end());

                bb[0] = min.x(); bb[1] = min.y(); bb[2] = min.z();
                bb[3] = max.x(); bb[4] = max.y(); bb[5] = max.z();

                send.isend(dest);

                current_file_idx++;
                active_workers++;
            }
            PROFILING_PRINT();
        }

        qems::QEMMesh mesh;
        mpi::PackedMessage msg(layout);

        while(active_workers > 0) {
            const int dest = mpi::sync_recv(msg); 
            const auto& recv_name = msg.get_buffer<char>(CSTM_TAG_NAME);
            const auto& recv_vertices = msg.get_buffer<float>(CSTM_TAG_VERT);
            const auto& recv_faces = msg.get_buffer<uint32_t>(CSTM_TAG_FACE);

            if (current_file_idx < files.size()) {
                const auto& file = files[current_file_idx];

                {
                    PROFILING_SCOPE("Sending-" + file.filename().string());

                    send.wait();

                    auto& msg = send.get_message();

                    auto& name = msg.get_buffer<char>(CSTM_TAG_NAME);
                    auto& bb = msg.get_buffer<double>(CSTM_TAG_BB);
                    auto& vertices = msg.get_buffer<float>(CSTM_TAG_VERT);
                    auto& faces = msg.get_buffer<uint32_t>(CSTM_TAG_FACE);

                    qems::import_mesh(file, metadata, vertices, faces);

                    LOG_DEBUG("{} - Imported {} with {} vertices, {} faces", 
                              pid, metadata.name,
                              vertices.size() / 3, 
                              faces.size() / 3);

                    name.clear();
                    name.assign(metadata.name.begin(), metadata.name.end());

                    bb[0] = min.x(); bb[1] = min.y(); bb[2] = min.z();
                    bb[3] = max.x(); bb[4] = max.y(); bb[5] = max.z();

                    send.isend(dest);

                    current_file_idx++;
                }
                PROFILING_PRINT();

            } else {
                mpi::sync_send(dest, {});
                active_workers--;
            }

            qems::row_data_to_mesh(recv_vertices, recv_faces, mesh);
            std::string out_name(recv_name.data(), recv_name.size());
            massert(OpenMesh::IO::write_mesh(mesh, "out/" + out_name), "Error in mesh export!");
        }
    }

    MPI_Finalize();
    return 0;
}
