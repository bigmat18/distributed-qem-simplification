#include <iostream>
#include <sys/time.h>
#include <unistd.h>

#include "mpi.h"

int main(int argc, char* argv[]) {
    int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    #pragma omp parallel
    {
        #pragma omp task
        {
            // - Iterater over files list
            // - Open a file and create MeshData
            // - Partition MeshData.mesh in submesh
            // - Build and add a task
        }

        #pragma omp task
        {
            // - send tasks
            // - receive task completed
        }

        #pragma omp task
        {
            // - check if a full file is completed
            // - if a file is completed merge it
            // - execute sequential refinement
            // - save the result
        }
    }

    MPI_Finalize();
	return 0;
}

