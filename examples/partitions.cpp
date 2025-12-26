#include <cstdint>
#include <cxxopts.hpp>
#include <unistd.h>
#include <queue>
#include <utils.hpp>

#include <qem_mesh.hpp>
#include <qem_utils.hpp>

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
    mesh.request_edge_status();
    mesh.request_face_status();
    mesh.request_halfedge_status();

    return 0;
} 
