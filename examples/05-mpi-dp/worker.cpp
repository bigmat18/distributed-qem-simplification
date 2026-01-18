#include <cstdint>
#include <sys/time.h>
#include <unistd.h>
#include <cxxopts.hpp>

#include <mpi.h>
#include <utils.hpp>

#include "qem_mesh.hpp"
#include "qem_simp.hpp"
#include "uniform_grid.hpp"
#include "mesh_import.hpp"
#include "async_send.hpp"
#include "sync_send_recv.hpp"
#include "message_layout.hpp"
#include "packed_message.hpp"

#define CSTM_TAG_BB 1
#define CSTM_TAG_VERT 2
#define CSTM_TAG_FACE 3
#define CSTM_TAG_NAME 4
#define CSTM_MESH 5

int main (int argc, char *argv[]) {

    int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    cxxopts::Options options("cli", "CLI app to test distributed mesh simplification");
    options.add_options()      
        ("i,input", "Input Folder", cxxopts::value<std::string>())
        ("n,meshes", "Num meshes", cxxopts::value<uint32_t>());
 
    options.parse_positional({"input"});
    auto result = options.parse(argc, argv);
 
    if(result.count("help")) { 
        printf("%s", options.help().c_str()); 
        return 0; 
    } 
 
    const std::string INPUT           = result["input"].as<std::string>();
    const uint32_t    NUM_MESHES      = result["meshes"].as<uint32_t>();

    massert(fs::exists("out") && fs::is_directory("out"), 
            "out folder does not exists");
    massert(fs::exists(INPUT) && fs::is_directory(INPUT), 
            "Input must be a valid folder");

	int pid, num_procs;
	MPI_Comm_size(MPI_COMM_WORLD,&num_procs); 
	MPI_Comm_rank(MPI_COMM_WORLD,&pid); 


    MPI_Finalize();
    return 0;
}
