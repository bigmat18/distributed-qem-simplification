#!/bin/bash

mkdir -p tests
ITERATIONS=2

# ==============================================================================
# 06-mpi-omp
# ==============================================================================
for N in 1 2 3 4 5 6 7; do
    # Bunny
    for i in $(seq 1 $ITERATIONS); do
        ./SRUN.sh release 06-mpi-omp -i "/home/m.giuntoni3/stanford/stanford_bunny.ply" -p 32 -nw "$N" -mt 32 -wt 32 -t 0.01 >> tests/${N}_mpi-omp-bunny.txt
    done

    # Armadillo
    for i in $(seq 1 $ITERATIONS); do
        ./SRUN.sh release 06-mpi-omp -i "/home/m.giuntoni3/stanford/stanford_armadillo.ply" -p 32 -nw "$N" -mt 32 -wt 32 -t 0.01 >> tests/${N}_mpi-omp-armadillo.txt
    done

    # Dragon
    for i in $(seq 1 $ITERATIONS); do
        ./SRUN.sh release 06-mpi-omp -i "/home/m.giuntoni3/stanford/stanford_dragon.ply" -p 32 -nw "$N" -mt 32 -wt 32 -t 0.01 >> tests/${N}_mpi-omp-dragon.txt
    done
    
    # Lucy
    for i in $(seq 1 $ITERATIONS); do
        ./SRUN.sh release 06-mpi-omp -i "/home/m.giuntoni3/stanford/stanford_lucy.ply" -p 32 -nw "$N" -mt 32 -wt 32 -t 0.01 >> tests/${N}_mpi-omp-lucy.txt
    done
done

# ==============================================================================
# 07-mpi
# ==============================================================================
for N in 8 16 32 64 128 160 192 224; do
    # Bunny
    for i in $(seq 1 $ITERATIONS); do
        ./SRUN.sh release 07-mpi -i "/home/m.giuntoni3/stanford/stanford_bunny.ply" -p 16 -nw $N -mt 32 -wt 1 -t 0.01 >> tests/${N}_07-mpi-bunny.txt
    done

    # Armadillo
    for i in $(seq 1 $ITERATIONS); do
        ./SRUN.sh release 07-mpi -i "/home/m.giuntoni3/stanford/stanford_armadillo.ply" -p 16 -nw $N -mt 32 -wt 1 -t 0.01 >> tests/${N}_07-mpi-armadillo.txt
    done

    # Dragon
    for i in $(seq 1 $ITERATIONS); do
        ./SRUN.sh release 07-mpi -i "/home/m.giuntoni3/stanford/stanford_dragon.ply" -p 16 -nw $N -mt 32 -wt 1 -t 0.01 >> tests/${N}_07-mpi-dragon.txt
    done

    # Lucy
    for i in $(seq 1 $ITERATIONS); do
        ./SRUN.sh release 07-mpi -i "/home/m.giuntoni3/stanford/stanford_lucy.ply" -p 16 -nw $N -mt 32 -wt 1 -t 0.01 >> tests/${N}_07-mpi-lucy.txt
    done
done

echo "All tests completed."