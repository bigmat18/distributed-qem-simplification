#include <cxxopts.hpp>
#include <mpi.h>
#include <print>
#include <utils.hpp>

#include "master.hpp"
#include "worker.hpp"

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

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    if (pid == 0) {
        const char* env = std::getenv("OMP_NUM_THREADS_MASTER");
        omp_set_num_threads(atoi(env));
        Main_Master(pid, num_procs, INPUT, START_PARTITIONS, NUM_MESHES, TARGET);
    } else {
        const char* env = std::getenv("OMP_NUM_THREADS_WORKER");
        omp_set_num_threads(atoi(env));
        Main_Worker(pid, num_procs, INPUT, START_PARTITIONS, NUM_MESHES, TARGET);
    }

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    if (pid == 0)
        std::println("time: {:.2f}", elapsed_time);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
