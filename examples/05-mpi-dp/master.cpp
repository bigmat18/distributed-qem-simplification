#include <cstdint>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <cxxopts.hpp>

#include <mpi.h>
#include <utils.hpp>

#include "profiling.hpp"
#include "qem_mesh.hpp"
#include "ug_row_data.hpp"
#include "mesh_import.hpp"

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


    std::vector<fs::path> files;
    for (const auto& file : fs::directory_iterator(INPUT)) {
        if (fs::is_regular_file(file.status()))
            files.push_back(file);
    }

    auto file = files[0];
    qems::MeshMetaData metadata;
    std::vector<float> vertices;
    std::vector<uint32_t> faces;
    mpi::UniformGridRow uniform_grid;

    {
        PROFILING_SCOPE("Test");

        {
            PROFILING_SCOPE("Import");
            qems::import_mesh("assets/stanford/stanford_lucy.ply", metadata, vertices, faces);
        }

        {
            PROFILING_SCOPE("UM-Building");
            uniform_grid = mpi::UniformGridRow(vertices, faces, metadata.min_coords, metadata.max_coords);
        }

        {
            PROFILING_SCOPE("Merge");
            uniform_grid.merge_cells(vertices, faces);
        }

        {
            PROFILING_SCOPE("Export");
            qems::export_mesh("out/" + metadata.name, vertices, faces); 
        }
    }
    PROFILING_PRINT();

    MPI_Finalize();
    return 0;
}
