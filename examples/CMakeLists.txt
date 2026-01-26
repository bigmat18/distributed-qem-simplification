cmake_minimum_required(VERSION 3.2)
project(qems-examples)

set(EXAMPLES_MP 
    01-sequential
    02-omp-uniform
    03-omp-octree
    04-ff-uniform
)

set(EXAMPLES_MPI
    05-mpi-omp
    06-mpi-omp
    07-mpi
)

foreach (subdir ${EXAMPLES_MP})
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


foreach (subdir ${EXAMPLES_MPI})
    set(SUBDIR_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})

    if (IS_DIRECTORY ${SUBDIR_PATH})

        file(GLOB CPP_FILES "${SUBDIR_PATH}/*.cpp")

        foreach (src ${CPP_FILES})
            get_filename_component(filename ${src} NAME_WE)
            set(TARGET_NAME "${subdir}_${filename}")

            add_executable(${TARGET_NAME} ${src})
            target_link_libraries(${TARGET_NAME} PUBLIC qems cxxopts)

            set_target_properties(${TARGET_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/${subdir}"
                OUTPUT_NAME "${filename}"
            )
        endforeach()

    else()
        message(STATUS "-- Directory ${subdir} non trovata, skipped")
    endif()
endforeach()
