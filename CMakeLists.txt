cmake_minimum_required(VERSION 3.26)
project(DistributedMeshSimplification CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LIBS_DIR "libs")
set(SRC_DIR "src")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

add_subdirectory(${LIBS_DIR}/cpp-utils-lib)
add_subdirectory(${LIBS_DIR}/eigen)
add_subdirectory(${LIBS_DIR}/cxxopts)

set(OPENMESH_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(OPENMESH_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${LIBS_DIR}/OpenMesh")

file(GLOB_RECURSE CPP_FILES ${SRC_DIR}/*.cpp)

function(configure_exe_target target_name)
  target_include_directories(${target_name}
    PUBLIC
      "${SRC_DIR}"
      "${LIBS_DIR}/eigen"
      SYSTEM "${LIBS_DIR}/OpenMesh/src"
      "${LIBS_DIR}/cxxopts/include"
  )

  target_link_libraries(${target_name}
    PUBLIC
      OpenMeshCore
      OpenMeshTools
      cxxopts
      OpenMP::OpenMP_CXX
      MPI::MPI_CXX
      stdc++exp
      cpp-utils-lib
  )

  target_compile_options(${target_name} PUBLIC 
    -enable-libstdcxx-backtrace=yes
    -Wno-deprecated-enum-enum-conversion
    -Wno-unused-result
    -Wno-deprecated-declarations
  )
endfunction()

foreach(src_file IN LISTS CPP_FILES)
  get_filename_component(exe_name "${src_file}" NAME_WE)
  add_executable(${exe_name} "${src_file}")
  configure_exe_target(${exe_name})
endforeach()
