#include <cstdint>
#include <cxxopts.hpp>
#include <unistd.h>
#include <vector>

#include <qem_mesh.hpp>
#include <qem_simp.hpp>
#include <utils.hpp>

int main(int argc, char **argv) {
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


    qems::QEMMesh mesh;
    massert(OpenMesh::IO::read_mesh(mesh, FILENAME), "Error in mesh import");

    LOG_INFO("{} successfully imported", FILENAME.c_str());
    mesh.request_vertex_status();
    mesh.request_edge_status();
    mesh.request_face_status();
    mesh.request_halfedge_status();
 
    {
        PROFILING_SCOPE("QEM-Simplification");

        std::vector<qems::QEMMesh::EdgeHandle> edges;
        {
            PROFILING_SCOPE("Pre-Proccessing");

            LOG_DEBUG("Init Vertices Quadric");
            {
                PROFILING_SCOPE("Init-Vertices-Quadratic");
                for (auto vh : mesh.vertices())
                    mesh.data(vh).Quadric = qems::compute_vertex_quadratic(mesh, vh);
            }

            LOG_DEBUG("Init Edges Quadric");
            {
                PROFILING_SCOPE("Init-Edges-Quadric");
                for (auto eh : mesh.edges()) {
                    auto heh = mesh.halfedge_handle(eh, 0);
                    auto v0 = mesh.from_vertex_handle(heh);
                    auto v1 = mesh.to_vertex_handle(heh);

                    Eigen::Matrix4d Q = mesh.data(v0).Quadric + mesh.data(v1).Quadric;
                    Eigen::Vector4d newV = qems::compute_new_best_vertex(mesh, eh, Q);

                    mesh.data(eh).Error = newV.transpose() * Q * newV;
                    mesh.data(eh).NewVertex = newV;
                    edges.push_back(eh);
                } 
            }
        }

        qems::QEMPriorityQueue pq(qems::QEMEdgeCompare(mesh), edges);

        LOG_DEBUG("Start processing");
        {
            PROFILING_SCOPE("Processing");
            qems::simplification(mesh, TARGET_FACES, mesh.n_faces(), pq); 
        }


        LOG_DEBUG("Start mesh cleanup");
        {
            PROFILING_SCOPE("Mesh Cleanup");
            mesh.garbage_collection();
        }
    }

    LOG_DEBUG("Mesh vertices: {}, edges: {}, faces: {}", 
              mesh.n_vertices(), mesh.n_edges(), mesh.n_faces());

    massert(OpenMesh::IO::write_mesh(mesh, "out/sequential.ply"), "Error in mesh export!");
    LOG_INFO("Mesh successfully exported!");

    PROFILING_PRINT();
    return 0;
} 
