cmake_minimum_required(VERSION 3.26)
project(DistributedMeshSimplification CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EXTERNAL_DIR "external")
set(EXAMPLE_DIR "examples")
set(INCLUDE_DIR "include")
set(SRC_DIR "src")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

set(CPPUTILS_BUILD_EXAMPLES   OFF CACHE BOOL "CPPUtils build examples" FORCE)
set(CPPUTILS_USE_OPENMP       ON  CACHE BOOL "CPPUtils enable OpenMP"  FORCE)

set(OPENMESH_BUILD_APPS       OFF CACHE BOOL "" FORCE)
set(OPENMESH_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)

set(BUILD_TESTING             OFF CACHE BOOL "Disable tests for all deps (including Eigen)" FORCE)
set(EIGEN_BUILD_DOC           OFF CACHE BOOL "Disable Eigen documentation"                  FORCE)
set(EIGEN_BUILD_PKGCONFIG     ON CACHE BOOL "Generate eigen3.pc for pkg-config"           FORCE)

add_subdirectory(${EXTERNAL_DIR}/cpp-utils-lib)
add_subdirectory(${EXTERNAL_DIR}/eigen)
add_subdirectory(${EXTERNAL_DIR}/cxxopts)
add_subdirectory(${EXTERNAL_DIR}/OpenMesh)

function(configure_exe_target target_name)
    target_include_directories(${target_name}
        PUBLIC
            ${INCLUDE_DIR}
            ${EXTERNAL_DIR}/cxxopts/include
        SYSTEM
            ${EXTERNAL_DIR}/eigen
            ${EXTERNAL_DIR}/OpenMesh/src
    )

    target_link_libraries(${target_name}
        PUBLIC
            OpenMeshCore
            OpenMeshTools
            cxxopts
            OpenMP::OpenMP_CXX
            MPI::MPI_CXX
            cpp-utils-lib
    )
endfunction()

file(GLOB_RECURSE EXAMPLES_FILES ${EXAMPLE_DIR}/*.cpp)
file(GLOB_RECURSE CPP_FILES ${SRC_DIR}/*.cpp)

foreach(file IN LISTS EXAMPLES_FILES)
    get_filename_component(exe_name ${file} NAME_WE)
    add_executable(${exe_name} ${file} ${CPP_FILES})
    configure_exe_target(${exe_name})
endforeach()
