cmake_minimum_required(VERSION 3.2)
project(qems-examples)

set(EXAMPLES 
    01-sequential
    02-omp-uniform
    03-omp-octree
    04-ff-uniform
    05-mpi-omp
    06-mpi-omp
    07-mpi
)

foreach (subdir ${EXAMPLES})
    set(SUBDIR_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
    set(MAIN_FILE ${SUBDIR_PATH}/main.cpp)

    if (IS_DIRECTORY ${SUBDIR_PATH} AND EXISTS ${MAIN_FILE})

        add_executable(${subdir} ${MAIN_FILE})
        target_link_libraries(${subdir} PUBLIC qems cxxopts)
        target_include_directories(${subdir} PUBLIC "${EXTERNAL_DIR}/fastflow")

    else()
        message(STATUS "-- main.cpp don't exists in ${subdir}, skipped")
    endif()
endforeach()

