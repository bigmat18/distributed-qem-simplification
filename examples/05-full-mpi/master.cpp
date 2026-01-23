#include <cstdint>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <cxxopts.hpp>

#include <mpi.h>
#include <utils.hpp>

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
        for (const auto& file : fs::directory_iterator(INPUT)) {
            if (fs::is_regular_file(file.status()))
                files.push_back(file);
        }

        auto file = files[0];
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

        mpi::AsyncSend async_sender(layout, num_procs-1);

        uint32_t current_idx = 0;
        auto& cells = uniform_grid.cells();

        for(int dest = 1; dest < num_procs; ++dest) {
            if(current_idx >= cells.size())
                continue;

            auto& cell = cells[current_idx];
            async_sender.wait();
            auto& msg = async_sender.get_message();

            msg.get_buffer<uint32_t>(CSTM_TAG_CELL_ID) = {current_idx, current_idx};
            msg.get_buffer<uint32_t>(CSTM_TAG_CELL_PART_LVL) = {START_PARTITIONS, START_PARTITIONS};

            float total_faces = static_cast<float>(faces.size()/3);
            uint32_t final_target = static_cast<uint32_t>(std::floor(total_faces * TARGET));
            msg.get_buffer<uint32_t>(CSTM_TAG_FINAL_TARGET) = { final_target };

            auto& bb = msg.get_buffer<double>(CSTM_TAG_BB);
            bb[0] = min.x(); bb[1] = min.y(); bb[2] = min.z();
            bb[3] = max.x(); bb[4] = max.y(); bb[5] = max.z();

            msg.get_buffer<char>(CSTM_TAG_NAME).clear();
            msg.get_buffer<char>(CSTM_TAG_NAME).assign(metadata.name.begin(), metadata.name.end());

            std::swap(msg.get_buffer<float>(CSTM_TAG_VERT), cell.vertices);
            std::swap(msg.get_buffer<uint32_t>(CSTM_TAG_FACE), cell.faces);
            std::swap(msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP), cell.indices_mapping);
            async_sender.isend(dest);
            current_idx++;
        }

        mpi::PackedMessage msg(layout);
        while(true) {
            const int dest = mpi::sync_recv(msg);
            const auto& recv_name = msg.get_buffer<char>(CSTM_TAG_NAME);
            const auto& recv_vertices = msg.get_buffer<float>(CSTM_TAG_VERT);
            const auto& recv_faces = msg.get_buffer<uint32_t>(CSTM_TAG_FACE);

            if (recv_vertices.empty()) {
                if (current_idx < cells.size()) {
                    auto& cell = cells[current_idx];
                    async_sender.wait();
                    auto& msg = async_sender.get_message();

                    msg.get_buffer<uint32_t>(CSTM_TAG_CELL_ID) = {current_idx, current_idx};
                    msg.get_buffer<uint32_t>(CSTM_TAG_CELL_PART_LVL) = {START_PARTITIONS, START_PARTITIONS};

                    float total_faces = static_cast<float>(faces.size()/3);
                    uint32_t final_target = static_cast<uint32_t>(std::floor(total_faces * TARGET));
                    msg.get_buffer<uint32_t>(CSTM_TAG_FINAL_TARGET) = { final_target };

                    auto& bb = msg.get_buffer<double>(CSTM_TAG_BB);
                    bb[0] = min.x(); bb[1] = min.y(); bb[2] = min.z();
                    bb[3] = max.x(); bb[4] = max.y(); bb[5] = max.z();

                    msg.get_buffer<char>(CSTM_TAG_NAME).clear();
                    msg.get_buffer<char>(CSTM_TAG_NAME).assign(metadata.name.begin(), metadata.name.end());

                    std::swap(msg.get_buffer<float>(CSTM_TAG_VERT), cell.vertices);
                    std::swap(msg.get_buffer<uint32_t>(CSTM_TAG_FACE), cell.faces);
                    std::swap(msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP), cell.indices_mapping);
                    async_sender.isend(dest);
                    current_idx++;
                }
            } else {
                std::string str_name(recv_name.data(), recv_name.size());
                qems::export_mesh("out/"+str_name, recv_vertices, recv_faces);

                for (int w = 1; w < num_procs; ++w)
                    mpi::sync_send(w, {});

                break;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
