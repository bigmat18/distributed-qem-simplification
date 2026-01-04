#include "logging.hpp"
#include "profiling.hpp"
#include <cmath>
#include <cstdint>
#include <cxxopts.hpp>
#include <omp.h>
#include <unistd.h>
#include <utils.hpp>

#include <qem_mesh.hpp>
#include <qem_simp.hpp>
#include <octree.hpp>

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
    mesh.request_vertex_colors();
    mesh.request_edge_status();
    mesh.request_face_status();
    mesh.request_halfedge_status();
 
    qems::Octree octree;
    {
        PROFILING_SCOPE("QEM-Simplification");

        {
            PROFILING_SCOPE("Pre-Processing");
            Eigen::Vector3d min;
            Eigen::Vector3d max;

            {
                PROFILING_SCOPE("Compute-Bounding-Box");
                qems::compute_bounding_box(mesh, min, max); 
            }

            uint32_t limit = mesh.n_vertices() / omp_get_max_threads();
            octree = qems::Octree(min, max, limit);
            #pragma omp declare reduction(                           \
                octree_merge : qems::Octree : omp_out.merge(omp_in)) \
                initializer(omp_priv = qems::Octree(omp_orig)        \
            )

            LOG_DEBUG("Start Octree building");

            PROFILING_LOCK();
            #pragma omp parallel reduction(octree_merge : octree)
            {
                PROFILING_SCOPE("Compute-Octree");
                {
                    #pragma omp for schedule(static) 
                    for (size_t i = 0; i < mesh.n_vertices(); ++i) {
                        auto vh = qems::QEMMesh::VertexHandle(i);
                        mesh.data(vh).Quadric = qems::compute_vertex_quadratic(mesh, vh);
                        uint32_t idx = octree.add_vertex(mesh, vh);
                        mesh.data(vh).NodeIdx = idx;
                    }
                }

                {
                    #pragma omp for schedule(static)  
                    for (size_t i = 0; i < mesh.n_edges(); ++i) {
                        auto eh = qems::QEMMesh::EdgeHandle(i);
                        auto heh = mesh.halfedge_handle(eh, 0);
                        auto vh0 = mesh.from_vertex_handle(heh);
                        auto vh1 = mesh.to_vertex_handle(heh);

                        size_t idx0 = octree.get_vertex_indices(mesh, vh0, min, max).w();
                        size_t idx1 = octree.get_vertex_indices(mesh, vh1, min, max).w();

                        if (idx0 == idx1) {
                            Eigen::Matrix4d Q = mesh.data(vh0).Quadric + mesh.data(vh1).Quadric;
                            Eigen::Vector4d newV = qems::compute_new_best_vertex(mesh, eh, Q);

                            mesh.data(eh).Error = newV.transpose() * Q * newV;
                            mesh.data(eh).NewVertex = newV;
                        } else {
                            mesh.data(vh0).Collasable = false;
                            mesh.data(vh1).Collasable = false;
                        }
                    }
                }

                {
                    #pragma omp for schedule(static)
                    for (size_t i = 0; i < mesh.n_edges(); ++i) {
                        auto eh = qems::QEMMesh::EdgeHandle(i);
                        octree.add_edge(mesh, eh);
                    }
                }

                {
                    #pragma omp for schedule(static)
                    for(size_t i = 0; i < mesh.n_faces(); i++) {
                        auto fh = qems::QEMMesh::FaceHandle(i);
                        octree.increment_collasable_faces(mesh, fh);
                    }
                }

            }
            PROFILING_UNLOCK();

            {
                PROFILING_SCOPE("Octree-Normalization");
                octree.normalize(mesh);
            }
        }

        LOG_DEBUG("Start Parallel QEM-Simplification");
        auto tree = octree.get_nodes();
 
        {
            PROFILING_SCOPE("Processing");

            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t j = 0; j < tree.size(); j++) { 
                if (!tree[j].is_leaf)
                    continue;

                auto& node = tree[j];
                uint32_t local_num_faces = node.collasable_faces;
                float total_faces = static_cast<float>(octree.total_collasable_faces());
                float cell_faces  = static_cast<float>(local_num_faces);
                
                float fraction = (total_faces > 0.0) ? (cell_faces / total_faces) : 0.0;
                float target_d = static_cast<float>(TARGET_FACES) * fraction;
                
                uint32_t local_target = static_cast<uint32_t>(std::floor(target_d));
                qems::QEMPriorityQueue pq(qems::QEMEdgeCompare(mesh), node.edges);

                qems::simplification(mesh, local_target, local_num_faces, pq);
            }
        }


        {
            PROFILING_SCOPE("Mesh-Cleanup");
            mesh.garbage_collection();
        }


        LOG_DEBUG("Parallel computation mesh vertices: {}, edges: {}, faces: {}", 
                  mesh.n_vertices(), mesh.n_edges(), mesh.n_faces());

        {
            PROFILING_SCOPE("Refinements");
            std::vector<qems::QEMMesh::EdgeHandle> edges;
            edges.reserve(mesh.n_edges());

            #pragma omp parallel
            {
                std::vector<qems::QEMMesh::EdgeHandle> local_edges;
                size_t n = mesh.n_edges();
                int num_threads = omp_get_num_threads();
                local_edges.reserve((n + num_threads - 1) / num_threads);

                #pragma omp for schedule(static) 
                for (size_t i = 0; i < mesh.n_vertices(); ++i) {
                    auto vh = qems::QEMMesh::VertexHandle(i);
                    mesh.data(vh).Quadric = qems::compute_vertex_quadratic(mesh, vh);
                }
                #pragma omp for schedule(static)  
                for (size_t i = 0; i < mesh.n_edges(); ++i) {
                    auto eh = qems::QEMMesh::EdgeHandle(i);
                    auto heh = mesh.halfedge_handle(eh, 0);
                    auto vh0 = mesh.from_vertex_handle(heh);
                    auto vh1 = mesh.to_vertex_handle(heh);

                    Eigen::Matrix4d Q = mesh.data(vh0).Quadric + mesh.data(vh1).Quadric;
                    Eigen::Vector4d newV = qems::compute_new_best_vertex(mesh, eh, Q);

                    mesh.data(eh).Error = newV.transpose() * Q * newV;
                    mesh.data(eh).NewVertex = newV;
                    local_edges.push_back(eh);
                }

                #pragma omp critical
                {
                    edges.insert(edges.end(),
                                 local_edges.begin(),
                                 local_edges.end());
                }
            }

            qems::QEMPriorityQueue pq(qems::QEMEdgeCompare(mesh), edges);
            qems::simplification(mesh, TARGET_FACES, mesh.n_faces(), pq);
            mesh.garbage_collection();
        }
    }

    //octree.export_mesh("out/wireframe.obj");

    LOG_DEBUG("Final computation mesh vertices: {}, edges: {}, faces: {}", 
              mesh.n_vertices(), mesh.n_edges(), mesh.n_faces());

    massert(OpenMesh::IO::write_mesh(mesh, "out/octree.ply"), "Error in mesh export!");
    LOG_INFO("Mesh successfully exported!");

    PROFILING_PRINT();
    return 0; 
}
