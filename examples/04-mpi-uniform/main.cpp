#include <cstdio>
#include <mpi.h>
#include <omp.h>
#include <string.h>
#include <print>

int main (int argc, char *argv[]){
    char message[20];
    int myrank;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) { /* process zero */
        strcpy(message, "Hello, there");
        MPI_Send(message, strlen(message) + 1, MPI_CHAR, 1, 99, MPI_COMM_WORLD);
    } else if (myrank == 1) { /* process one */
        MPI_Recv(message, 20, MPI_CHAR, 0, 99, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_CHAR, &count);
        printf("received: %s (MPI_CHAR received %d)\n", message, count);
    }

    MPI_Finalize();
    return 0;
	return 0;
}

