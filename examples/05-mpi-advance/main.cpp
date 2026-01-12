#include <cstdint>
#include <sys/time.h>
#include <unistd.h>

#include <cxxopts.hpp>

#include "mesh_import.hpp"
#include "mpi.h"

#include "async_send.hpp"
#include "sync_send_recv.hpp"
#include "message_layout.hpp"
#include "packed_message.hpp"

#define CSTM_TAG_BB 1
#define CSTM_TAG_VERT 2
#define CSTM_TAG_FACE 3
#define CSTM_MESH 4

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

    mpi::MessageLayout layout(CSTM_MESH);
    layout 
     .add_buffer<double>(CSTM_TAG_BB)
     .add_buffer<float>(CSTM_TAG_VERT)
     .add_buffer<uint32_t>(CSTM_TAG_FACE);

    if (pid == 0) {
        mpi::AsyncSend send(layout);

        int dest = 1;
        for (const auto& file : fs::directory_iterator(INPUT)) {
            if (!fs::is_regular_file(file.status()))
                continue;

            qems::MeshData load_data;
            qems::import_mesh(file, load_data);
            LOG_INFO("{} - Imported {} vertuces, {} faces", 
                     pid,
                     load_data.row_vertices.size() / 3, 
                     load_data.row_faces.size() / 3);

            const auto& min = load_data.min_coords;
            const auto& max = load_data.max_coords;

            std::vector<double> bb(6);
            bb[0] = min.x(); bb[1] = min.y(); bb[2] = min.z();
            bb[3] = max.x(); bb[4] = max.y(); bb[5] = max.z();


            send.isend(dest, {
                {CSTM_TAG_BB, std::move(bb)},
                {CSTM_TAG_VERT, std::move(load_data.row_vertices)},
                {CSTM_TAG_FACE, std::move(load_data.row_faces)}
            });

            dest++;
            if (dest >= num_procs) dest = 1;
        }
        send.wait();

        mpi::PackedMessage end_msg = mpi::PackedMessage(mpi::MessageLayout());
        for (int w = 1; w < num_procs; w++)
            mpi::sync_send(w, end_msg);

    } else {
        mpi::PackedMessage msg(layout);
        auto& bb = msg.get_buffer<double>(CSTM_TAG_BB);
        auto& vertices = msg.get_buffer<float>(CSTM_TAG_VERT);
        auto& faces = msg.get_buffer<uint32_t>(CSTM_TAG_FACE);

        while (true) {
            if (!mpi::sync_recv(0, msg)) 
                break;

            LOG_INFO("{} - Recived {} vertuces, {} faces", 
                     pid,
                     vertices.size() / 3, 
                     faces.size() / 3);
        }
    }

    MPI_Finalize();
	return 0;
}

