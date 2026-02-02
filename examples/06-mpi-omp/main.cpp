#include <cstdlib>
#include <cxxopts.hpp>

#include <mpi.h>
#include <omp.h>
#include <utils.hpp>

#include "utils.hpp"

#include "master_v1.hpp"
#include "master_v2.hpp"
#include "worker.hpp"

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

    const uint32_t GLOBAL_PARTITIONS = next_pow(std::ceil(std::cbrt(static_cast<float>(num_procs-1))));

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    if (pid == 0) {
        const char* env = std::getenv("OMP_NUM_THREADS_MASTER");
        int num_threads = atoi(env);
        omp_set_num_threads(num_threads);
        if (!fs::is_directory(INPUT))
            Main_Master_V1(pid, num_procs, INPUT, GLOBAL_PARTITIONS, NUM_MESHES, TARGET);
        else
            Main_Master_V2(pid, num_procs, INPUT, GLOBAL_PARTITIONS, NUM_MESHES, TARGET);
    } else {
        const char* env = std::getenv("OMP_NUM_THREADS_WORKER");
        int num_threads = atoi(env);
        omp_set_num_threads(num_threads);
        Main_Worker(pid, num_procs, INPUT, GLOBAL_PARTITIONS, START_PARTITIONS, NUM_MESHES, TARGET);
    }

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    MPI_Barrier(MPI_COMM_WORLD);
    if (pid == 0)
        std::cout << "time: " << std::fixed << std::setprecision(2) << elapsed_time << std::endl;

    MPI_Finalize();
    return 0;
}
