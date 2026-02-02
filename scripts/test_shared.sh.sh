#!/bin/bash
mkdir -p tests

ITERATIONS=1
export OMP_NUM_THREADS=1 

echo "Running 01-sequential"
for iter in $(seq 1 $ITERATIONS); do
    if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/01-sequential /home/m.giuntoni3/stanford/stanford_lucy.ply -n 280557 >> tests/sequential-lucy.txt 2>&1; then
        echo "Iteration $iter for 01-sequential lucy: OK"
    else
        echo "Iteration $iter for 01-sequential lucy: FAILED" >&2
    fi

    if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/01-sequential /home/m.giuntoni3/stanford/stanford_dragon.ply -n 72189 >> tests/sequential-dragon.txt 2>&1; then
        echo "Iteration $iter for 01-sequential dragon: OK"
    else
        echo "Iteration $iter for 01-sequential dragon: FAILED" >&2
    fi

    if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/01-sequential /home/m.giuntoni3/stanford/stanford_bunny.ply -n 694 >> tests/sequential-bunny.txt 2>&1; then
        echo "Iteration $iter for 01-sequential bunny: OK"
    else
        echo "Iteration $iter for 01-sequential bunny: FAILED" >&2
    fi

    if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/01-sequential /home/m.giuntoni3/stanford/stanford_armadillo.ply -n 3459 >> tests/sequential-armadillo.txt 2>&1; then
        echo "Iteration $iter for 01-sequential armadillo: OK"
    else
        echo "Iteration $iter for 01-sequential armadillo: FAILED" >&2
    fi
done

for N in 2 4 8 16 32; do
    echo "Starting tests for N=$N threads"
    export OMP_NUM_THREADS=$N

    # # Program 1: 02-omp-uniform-nored
    # echo "Running 02-omp-uniform-nored for N=$N"
    # for iter in $(seq 1 $ITERATIONS); do
    #     if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/02-omp-uniform /home/m.giuntoni3/stanford/stanford_lucy.ply -p 16 -n 280557 -t $N >> tests/$N-omp-uniform-nored-lucy.txt 2>&1; then
    #         echo "Iteration $iter for omp-uniform lucy: OK"
    #     else
    #         echo "Iteration $iter for omp-uniform lucy: FAILED" >&2
    #     fi

    #     if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/02-omp-uniform /home/m.giuntoni3/stanford/stanford_dragon.ply -p 16 -n 72189 -t $N >> tests/$N-omp-uniform-nored-dragon.txt 2>&1; then
    #         echo "Iteration $iter for omp-uniform dragon: OK"
    #     else
    #         echo "Iteration $iter for omp-uniform dragon: FAILED" >&2
    #     fi

    #     if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/02-omp-uniform /home/m.giuntoni3/stanford/stanford_bunny.ply -p 16 -n 694 -t $N >> tests/$N-omp-uniform-nored-bunny.txt 2>&1; then
    #         echo "Iteration $iter for omp-uniform bunny: OK"
    #     else
    #         echo "Iteration $iter for omp-uniform bunny: FAILED" >&2
    #     fi

    #     if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/02-omp-uniform /home/m.giuntoni3/stanford/stanford_armadillo.ply -p 16 -n 3459 -t $N >> tests/$N-omp-uniform-nored-armadillo.txt 2>&1; then
    #         echo "Iteration $iter for omp-uniform armadillo: OK"
    #     else
    #         echo "Iteration $iter for omp-uniform armadillo: FAILED" >&2
    #     fi
    # done

    Program 1: 02-omp-uniform
    echo "Running 02-omp-uniform for N=$N"
    for iter in $(seq 1 $ITERATIONS); do
        if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/02-omp-uniform /home/m.giuntoni3/stanford/stanford_lucy.ply -p 16 -n 280557 -t $N >> tests/$N-omp-uniform-lucy.txt 2>&1; then
            echo "Iteration $iter for omp-uniform lucy: OK"
        else
            echo "Iteration $iter for omp-uniform lucy: FAILED" >&2
        fi

        if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/02-omp-uniform /home/m.giuntoni3/stanford/stanford_dragon.ply -p 16 -n 72189 -t $N >> tests/$N-omp-uniform-dragon.txt 2>&1; then
            echo "Iteration $iter for omp-uniform dragon: OK"
        else
            echo "Iteration $iter for omp-uniform dragon: FAILED" >&2
        fi

        if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/02-omp-uniform /home/m.giuntoni3/stanford/stanford_bunny.ply -p 16 -n 694 -t $N >> tests/$N-omp-uniform-bunny.txt 2>&1; then
            echo "Iteration $iter for omp-uniform bunny: OK"
        else
            echo "Iteration $iter for omp-uniform bunny: FAILED" >&2
        fi

        if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/02-omp-uniform /home/m.giuntoni3/stanford/stanford_armadillo.ply -p 16 -n 3459 -t $N >> tests/$N-omp-uniform-armadillo.txt 2>&1; then
            echo "Iteration $iter for omp-uniform armadillo: OK"
        else
            echo "Iteration $iter for omp-uniform armadillo: FAILED" >&2
        fi
    done

    Program 2: 03-omp-octree
    echo "Running 03-omp-octree for N=$N"
    for iter in $(seq 1 $ITERATIONS); do
        if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/03-omp-octree /home/m.giuntoni3/stanford/stanford_lucy.ply -n 280557 -t $N >> tests/$N-omp-octree-lucy.txt 2>&1; then
            echo "Iteration $iter for omp-octree lucy: OK"
        else
            echo "Iteration $iter for omp-octree lucy: FAILED" >&2
        fi

        if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/03-omp-octree /home/m.giuntoni3/stanford/stanford_dragon.ply -n 72189 -t $N >> tests/$N-omp-octree-dragon.txt 2>&1; then
            echo "Iteration $iter for omp-octree dragon: OK"
        else
            echo "Iteration $iter for omp-octree dragon: FAILED" >&2
        fi

        if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/03-omp-octree /home/m.giuntoni3/stanford/stanford_bunny.ply -n 694 -t $N >> tests/$N-omp-octree-bunny.txt 2>&1; then
            echo "Iteration $iter for omp-octree bunny: OK"
        else
            echo "Iteration $iter for omp-octree bunny: FAILED" >&2
        fi

        if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/03-omp-octree /home/m.giuntoni3/stanford/stanford_armadillo.ply -n 3459 -t $N >> tests/$N-omp-octree-armadillo.txt 2>&1; then
            echo "Iteration $iter for omp-octree armadillo: OK"
        else
            echo "Iteration $iter for omp-octree armadillo: FAILED" >&2
        fi
    done

    # Program 4: 04-ff-uniform
    echo "Running 04-ff-uniform for N=$N"
    for iter in $(seq 1 $ITERATIONS); do
        if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/04-ff-uniform /home/m.giuntoni3/stanford/stanford_lucy.ply -p 16 -n 280557 -t $N >> tests/$N-ff-uniform-lucy.txt 2>&1; then
            echo "Iteration $iter for ff-uniform lucy: OK"
        else
            echo "Iteration $iter for ff-uniform lucy: FAILED" >&2
        fi

        if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/04-ff-uniform /home/m.giuntoni3/stanford/stanford_dragon.ply -p 16 -n 72189 -t $N >> tests/$N-ff-uniform-dragon.txt 2>&1; then
            echo "Iteration $iter for ff-uniform dragon: OK"
        else
            echo "Iteration $iter for ff-uniform dragon: FAILED" >&2
        fi

        if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/04-ff-uniform /home/m.giuntoni3/stanford/stanford_bunny.ply -p 16 -n 694 -t $N >> tests/$N-ff-uniform-bunny.txt 2>&1; then
            echo "Iteration $iter for ff-uniform bunny: OK"
        else
            echo "Iteration $iter for ff-uniform bunny: FAILED" >&2
        fi

        if time -p srun --cpus-per-task=32 --ntasks=1 build/Release/examples/04-ff-uniform /home/m.giuntoni3/stanford/stanford_armadillo.ply -p 16 -n 3459 -t $N >> tests/$N-ff-uniform-armadillo.txt 2>&1; then
            echo "Iteration $iter for ff-uniform armadillo: OK"
        else
            echo "Iteration $iter for ff-uniform armadillo: FAILED" >&2
        fi
    done
done

echo "All tests completed."
# salloc --nodes=8 --ntasks=32 --time=00:30:00
