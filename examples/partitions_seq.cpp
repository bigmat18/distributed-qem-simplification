#include "logging.hpp"
#include "profiling.hpp"
#include <cstddef>
#include <cstdint>
#include <cxxopts.hpp>
#include <unistd.h>
#include <queue>
#include <utils.hpp>

#include <qem_mesh.hpp>
#include <qem_utils.hpp>
#include <uniform_grid.hpp>
#include <format_utils.hpp>

using Mesh = QEMMesh;

int main(int argc, char **argv) {
    massert(argc > 1, "Need [input file]");

    cxxopts::Options options("cli", "CLI app to test distributed mesh simplification");
    options.add_options()      
        ("i,filename", "Input filename list", cxxopts::value<std::string>())
        ("n,target", "Target faces", cxxopts::value<uint32_t>());

    options.parse_positional({"filename"});
    auto result = options.parse(argc, argv);

    if(result.count("help")) {
        printf("%s", options.help().c_str()); 
        return 0;
    }

    massert(result.count("filename") >= 1, "Need [input filename]");
    const std::string FILENAME        = result["filename"].as<std::string>();
    const uint32_t    TARGET_FACES    = result["target"].as<uint32_t>();

    Mesh mesh;
    massert(OpenMesh::IO::read_mesh(mesh, FILENAME), "Error in mesh import");
    LOG_INFO("{} successfully imported", FILENAME.c_str());
    mesh.request_vertex_status();
    mesh.request_vertex_colors();
    mesh.request_edge_status();
    mesh.request_face_status();
    mesh.request_halfedge_status();
    

    {
        PROFILING_SCOPE("Computation");
        Eigen::Vector3f min;
        Eigen::Vector3f max;

        ComputeBoundingBox(mesh, min, max); 
        LOG_INFO("Bounding Box: {}, {}", min, max);

        UniformGrid<2> uniform_grid(min, max);

        {
            PROFILING_SCOPE("UniformGrid Building");

            for (size_t i = 0; i < mesh.n_vertices(); ++i) {
                auto vh = QEMMesh::VertexHandle(i);
                uniform_grid.add_vertex(mesh, vh);
            }

            for (size_t i = 0; i < mesh.n_edges(); ++i) {
                auto eh = QEMMesh::EdgeHandle(i);
                uniform_grid.add_edge(mesh, eh);
            }
        }
    }

    PROFILING_PRINT();

    OpenMesh::IO::Options wopt;
    wopt += OpenMesh::IO::Options::VertexColor;

    //massert(OpenMesh::IO::write_mesh(mesh, "out/out.ply", wopt), "Error in mesh export!");
    //LOG_INFO("Mesh successfully exported!");
    return 0;
} 
