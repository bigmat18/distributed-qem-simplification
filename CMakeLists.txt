cmake_minimum_required(VERSION 3.2)
project(DistributedMeshSimplification CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EXTERNAL_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(INCLUDE_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR         ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(EXAMPLES_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/examples)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

set(CPPUTILS_BUILD_EXAMPLES   OFF CACHE BOOL "CPPUtils build examples" FORCE)
set(CPPUTILS_USE_OPENMP       ON  CACHE BOOL "CPPUtils enable OpenMP"  FORCE)

set(OPENMESH_BUILD_APPS       OFF CACHE BOOL "" FORCE)
set(OPENMESH_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)

set(BUILD_TESTING             OFF CACHE BOOL "Disable tests for all deps (including Eigen)" FORCE)
set(EIGEN_BUILD_DOC           OFF CACHE BOOL "Disable Eigen documentation"                  FORCE)

add_subdirectory(${EXTERNAL_DIR}/cpp-utils-lib)
add_subdirectory(${EXTERNAL_DIR}/eigen)
add_subdirectory(${EXTERNAL_DIR}/OpenMesh)
add_subdirectory(${EXTERNAL_DIR}/cxxopts)

set(TARGET_LIB_NAME qems)
file(GLOB_RECURSE CPP_FILES ${SRC_DIR}/*.cpp)
add_library(${TARGET_LIB_NAME} ${CPP_FILES})

target_include_directories(${TARGET_LIB_NAME} 
    PUBLIC 
        ${INCLUDE_DIR}
    SYSTEM
        ${EXTERNAL_DIR}/eigen
        ${EXTERNAL_DIR}/OpenMesh/src
)
target_link_libraries(${TARGET_LIB_NAME} 
    PUBLIC
        OpenMeshCore
        OpenMeshTools
        OpenMP::OpenMP_CXX
        MPI::MPI_CXX
        cpp-utils-lib 
)

add_subdirectory(${EXAMPLES_DIR})
