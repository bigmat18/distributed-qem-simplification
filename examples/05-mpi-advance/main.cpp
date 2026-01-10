#include <algorithm>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>

#include <cxxopts.hpp>

#include "async_send.hpp"
#include "logging.hpp"
#include "mesh_import.hpp"
#include "mpi.h"

#define CSTM_TAG_END 0
#define CSTM_TAG_BB 1
#define CSTM_TAG_VERT 2
#define CSTM_TAG_FACE 3

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
        mpi::AsyncSend message;
        message
         .add_buffer<double>(CSTM_TAG_BB)
         .add_buffer<float>(CSTM_TAG_VERT)
         .add_buffer<uint32_t>(CSTM_TAG_FACE);

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


            message.isend(dest, {
                {CSTM_TAG_BB, std::move(bb)},
                {CSTM_TAG_VERT, std::move(load_data.row_vertices)},
                {CSTM_TAG_FACE, std::move(load_data.row_faces)}
            });
            dest++;
            if (dest >= num_procs) dest = 1;
        }
            
        message.wait();

        double end[6];
        MPI_Request request;
        for (int w = 1; w < num_procs; ++w) {
            MPI_Isend(&end[0], 6, MPI_DOUBLE, w, CSTM_TAG_END, MPI_COMM_WORLD, &request);
        }

    } else {
        double vec_buffer[6];
        MPI_Status status;
        int count;
        qems::MeshData data;
        auto& min = data.min_coords;
        auto& max = data.max_coords;

        while (true) {
            data.row_faces.clear();
            data.row_vertices.clear();

            MPI_Recv(&vec_buffer, 6, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == CSTM_TAG_END || status.MPI_TAG != CSTM_TAG_BB)
                break;

            min.x() = vec_buffer[0]; min.y() = vec_buffer[1]; min.z() = vec_buffer[2];
            max.x() = vec_buffer[3]; max.y() = vec_buffer[4]; max.z() = vec_buffer[5];

            MPI_Probe(0, CSTM_TAG_VERT, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_FLOAT, &count);
            data.row_vertices.resize(count);
            MPI_Recv(data.row_vertices.data(), count, MPI_FLOAT, 0, CSTM_TAG_VERT, MPI_COMM_WORLD, &status);

            MPI_Probe(0, CSTM_TAG_FACE, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_UNSIGNED, &count);
            data.row_faces.resize(count);
            MPI_Recv(data.row_faces.data(), count, MPI_UNSIGNED, 0, CSTM_TAG_FACE, MPI_COMM_WORLD, &status);

            LOG_INFO("{} - Recived {} vertuces, {} faces", 
                     pid,
                     data.row_vertices.size() / 3, 
                     data.row_faces.size() / 3);
        }
    }

    MPI_Finalize();
	return 0;
}

