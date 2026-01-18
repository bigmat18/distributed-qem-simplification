cmake_minimum_required(VERSION 3.2)
project(qems-examples)

set(EXAMPLES_MP 
    01-sequential
    02-omp-uniform
    03-omp-octree
    04-mpi-base
)

set(EXAMPLES_MPI
    05-mpi-dp
)

foreach (subdir ${EXAMPLES_MP})
    set(SUBDIR_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
    set(MAIN_FILE ${SUBDIR_PATH}/main.cpp)

    if (IS_DIRECTORY ${SUBDIR_PATH} AND EXISTS ${MAIN_FILE})

        add_executable(${subdir} ${MAIN_FILE})
        target_link_libraries(${subdir} PUBLIC qems cxxopts)

    else()
        message(STATUS "-- main.cpp don't exists in ${subdir}, skipped")
    endif()
endforeach()


foreach (subdir ${EXAMPLES_MPI})
    set(SUBDIR_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})

    if (IS_DIRECTORY ${SUBDIR_PATH})

        file(GLOB CPP_FILES "${SUBDIR_PATH}/*.cpp")

        foreach (src ${CPP_FILES})
            get_filename_component(exec_name ${src} NAME_WE)
            add_executable(${exec_name} ${src})
            target_link_libraries(${exec_name} PUBLIC qems cxxopts)
            set_target_properties(${exec_name} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY
                    ${CMAKE_BINARY_DIR}/examples/${subdir}
            )
        endforeach()

    else()
        message(STATUS "-- Directory ${subdir} non trovata, skipped")
    endif()
endforeach()
