#include "logging.hpp"
#include <cstdint>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <cxxopts.hpp>

#include <mpi.h>
#include <utils.hpp>

#include <qem_mesh.hpp>
#include <qem_simp.hpp>
#include <qem_mesh.hpp>
#include <ug_row_data.hpp>
#include <mesh_import.hpp>

#define CSTM_TAG_BB 1
#define CSTM_TAG_VERT 2
#define CSTM_TAG_FACE 3
#define CSTM_TAG_NAME 4
#define CSTM_MESH 5

inline int next_step(int n) {
    if (n == 1) return 0;
    return (n % 2 != 0) ? (n - 1) / 2 : (n > 4 ? n / 2 + 1 : 1);
}

int main (int argc, char *argv[]) {

    int provided;
	//MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    cxxopts::Options options("cli", "CLI app to test distributed mesh simplification");
    options.add_options()      
        ("i,input", "Input Folder", cxxopts::value<std::string>())
        ("n,meshes", "Num meshes", cxxopts::value<uint32_t>());
 
    options.parse_positional({"input"});
    auto result = options.parse(argc, argv);
 
    if(result.count("help")) { 
        printf("%s", options.help().c_str()); 
        return 0; 
    } 
 
    const std::string INPUT           = result["input"].as<std::string>();
    const uint32_t    NUM_MESHES      = result["meshes"].as<uint32_t>();

    massert(fs::exists("out") && fs::is_directory("out"), 
            "out folder does not exists");
    massert(fs::exists(INPUT) && fs::is_directory(INPUT), 
            "Input must be a valid folder");

	//int pid, num_procs;
	//MPI_Comm_size(MPI_COMM_WORLD,&num_procs); 
	//MPI_Comm_rank(MPI_COMM_WORLD,&pid); 


    std::vector<fs::path> files;
    for (const auto& file : fs::directory_iterator(INPUT)) {
        if (fs::is_regular_file(file.status()))
            files.push_back(file);
    }

    auto file = files[0];
    qems::MeshMetaData metadata;
    std::vector<float> vertices;
    std::vector<uint32_t> faces;

    mpi::UniformGridRow uniform_grid;
    qems::QEMMesh mesh;

    mesh.request_vertex_status();
    mesh.request_edge_status();
    mesh.request_face_status();
    mesh.request_halfedge_status();
    mesh.request_vertex_normals();
    mesh.request_face_normals();

    const uint32_t TARGET_FACES = 10000;
    {
        PROFILING_SCOPE("Test");

        {
            PROFILING_SCOPE("Import");
            qems::import_mesh(file, metadata, vertices, faces);
        }


        uint32_t partitions = 8;
        while(partitions > 0 && (faces.size() / 3) > TARGET_FACES) {

            uniform_grid = mpi::UniformGridRow(
                vertices, faces, metadata.min_coords, 
                metadata.max_coords, partitions
            );

            uint32_t current_idx = 0;
            for (auto& cell : uniform_grid) {
                qems::row_data_to_mesh(cell.vertices, cell.faces, cell.indices_mapping, mesh);
                mesh.update_normals();

                std::vector<qems::QEMMesh::EdgeHandle> edges;

                for (auto vh : mesh.vertices()) {
                    auto coords = mesh.point(vh);
                    mesh.data(vh).Quadric = qems::compute_vertex_quadratic(mesh, vh);
                    mesh.data(vh).NodeIdx = uniform_grid.get_vertex_index({coords[0], coords[1], coords[2]});
                }

                for (auto eh : mesh.edges()) {
                    auto heh = mesh.halfedge_handle(eh, 0);
                    auto vh0 = mesh.from_vertex_handle(heh);
                    auto vh1 = mesh.to_vertex_handle(heh);

                    uint32_t idx0 = mesh.data(vh0).NodeIdx;
                    uint32_t idx1 = mesh.data(vh1).NodeIdx;

                    if (idx0 != idx1) {
                        mesh.data(vh0).Collasable = false;
                        mesh.data(vh1).Collasable = false;
                    } 
                } 

                for (auto eh : mesh.edges()) {
                    auto heh = mesh.halfedge_handle(eh, 0);
                    auto vh0 = mesh.from_vertex_handle(heh);
                    auto vh1 = mesh.to_vertex_handle(heh);

                    if (mesh.data(vh0).Collasable && mesh.data(vh1).Collasable) {
                        Eigen::Matrix4d Q = mesh.data(vh0).Quadric + mesh.data(vh1).Quadric;
                        Eigen::Vector4d newV = qems::compute_new_best_vertex(mesh, eh, Q);

                        mesh.data(eh).Error = newV.transpose() * Q * newV;
                        mesh.data(eh).NewVertex = newV;
                        edges.push_back(eh);
                    }
                }

                uint32_t collasable_faces = 0;
                for (auto fh : mesh.faces()) {
                    bool end = false;
                    for (auto fv_it = mesh.cfv_iter(fh); fv_it.is_valid(); ++fv_it) {
                        auto vh = *fv_it;
                        if (!mesh.data(vh).Collasable) {
                            end = true;
                            break;
                        }
                    }
                    if (end) continue;
                    collasable_faces++;
                }

                auto pq = qems::QEMPriorityQueue(qems::QEMEdgeCompare(mesh), edges);

                float total_faces = static_cast<float>(uniform_grid.total_collasable_faces());
                float fraction = (total_faces > 0.0) ? (collasable_faces / total_faces) : 0.0;
                float target_d = static_cast<float>(TARGET_FACES) * fraction;
                uint32_t local_target = static_cast<uint32_t>(std::floor(target_d));

                //LOG_INFO("partitions: {}, faces: {}, total_faces: {}, collasable_faces: {}, target: {}",
                         //current_idx, cell.faces.size()/3, total_faces, collasable_faces, local_target);

                qems::simplification(mesh, local_target, collasable_faces, pq);
                mesh.garbage_collection();
                qems::mesh_to_row_data(mesh, cell.vertices, cell.faces, cell.indices_mapping);
                current_idx++;
            }

            uniform_grid.merge_cells(vertices, faces);
            partitions = next_step(partitions);
        }

        {
            PROFILING_SCOPE("Export");
            qems::export_mesh("out/" + metadata.name, vertices, faces); 
        }
    }
    PROFILING_PRINT();

    //MPI_Finalize();
    return 0;
}
