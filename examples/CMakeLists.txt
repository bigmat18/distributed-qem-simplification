cmake_minimum_required(VERSION 3.26)
project(qems-examples)

set(EXAMPLES 
    01-sequential
    02-omp-uniform
    03-omp-octree
    04-mpi-base
    05-mpi-advance
)

foreach (subdir ${EXAMPLES})
    set(SUBDIR_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
    set(MAIN_FILE ${SUBDIR_PATH}/main.cpp)

    if (IS_DIRECTORY ${SUBDIR_PATH} AND EXISTS ${MAIN_FILE})

        add_executable(${subdir} ${MAIN_FILE})
        target_link_libraries(${subdir} PUBLIC qems cxxopts)

    else()
        message(STATUS "-- main.cpp don't exists in ${subdir}, skipped")
    endif()
endforeach()
