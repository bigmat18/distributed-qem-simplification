#include <cstdint>
#include <sys/time.h>
#include <unistd.h>
#include <cxxopts.hpp>

#include <mpi.h>
#include <utils.hpp>

#include "utils.hpp"

inline void Main_Worker(int pid, int num_procs,
                        const std::string INPUT,
                        const uint32_t GLOBAL_PARTITIONS,
                        const uint32_t START_PARTITIONS,
                        uint32_t NUM_MESHES,
                        const float TARGET)
{
    mpi::MessageLayout layout = get_layout();
    mpi::PackedMessage msg(layout);
    auto& id = msg.get_element<uint32_t>(CSTM_TAG_CELL_ID);
    auto& name = msg.get_element<char>(CSTM_TAG_NAME);
    auto& bb = msg.get_element<double>(CSTM_TAG_BB);
    auto& vertices = msg.get_buffer<float>(CSTM_TAG_VERT);
    auto& faces = msg.get_buffer<uint32_t>(CSTM_TAG_FACE);
    auto& idx_mapping = msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP);

    Eigen::Vector3d min, max;
    while (true) {
        if (mpi::sync_recv(msg, 0) == -1) 
            break;

        std::string str_name(name.data(), name.size());

        min.x() = bb[0]; min.y() = bb[1]; min.z() = bb[2]; 
        max.x() = bb[3]; max.y() = bb[4]; max.z() = bb[5];

        Eigen::Vector3d local_min, local_max;
        auto local_bb = get_cell_bb(min, max, GLOBAL_PARTITIONS, id[0]);
        local_min = std::move(local_bb.first);
        local_max = std::move(local_bb.second);

        qems::QEMMesh mesh;
        qems::UniformGrid uniform_grid;

        mesh.request_vertex_status();
        mesh.request_edge_status();
        mesh.request_face_status();
        mesh.request_halfedge_status();
        mesh.request_vertex_normals();
        mesh.request_face_normals();

        {
            PROFILING_SCOPE("PID:"+ std::to_string(pid) +",Mesh:" + str_name);
            qems::row_data_to_mesh(vertices, faces, idx_mapping, mesh);
            mesh.update_normals();

            const uint32_t TARGET_FACES = static_cast<uint32_t>(mesh.n_faces() * TARGET);
            LOG_DEBUG("{} - Received {} with {} vertices, {} faces, target {}", 
                      pid, str_name,
                      mesh.n_vertices(), 
                      mesh.n_faces(),
                      TARGET_FACES);


            uint32_t subdivision = START_PARTITIONS;
            while (subdivision > 0 && mesh.n_faces() > TARGET_FACES) {
                uniform_grid = qems::UniformGrid(min, max, subdivision);

                #pragma omp declare reduction(                                      \
                    uniform_grid_merge : qems::UniformGrid : omp_out.merge(omp_in)) \
                    initializer(omp_priv = qems::UniformGrid(omp_orig))


                #pragma omp parallel reduction(uniform_grid_merge : uniform_grid)
                {
                    #pragma omp for schedule(static) 
                    for (size_t i = 0; i < mesh.n_vertices(); ++i) {
                        auto vh = qems::QEMMesh::VertexHandle(i);
                        mesh.data(vh).Quadric = qems::compute_vertex_quadratic(mesh, vh);
                        mesh.data(vh).NodeIdx = uniform_grid.add_vertex(mesh, vh);
                        mesh.data(vh).Collasable = true;
                    }

                    #pragma omp for schedule(static)  
                    for (size_t i = 0; i < mesh.n_edges(); ++i) {
                        auto eh = qems::QEMMesh::EdgeHandle(i);
                        auto heh = mesh.halfedge_handle(eh, 0);
                        auto vh0 = mesh.from_vertex_handle(heh);
                        auto vh1 = mesh.to_vertex_handle(heh);

                        auto coords0 = mesh.point(vh0);
                        auto coords1 = mesh.point(vh1);
                        uint32_t global_idx0 = mpi::UniformGridRow::get_vertex_index(
                            {coords0[0], coords0[1], coords0[2]}, min, max, GLOBAL_PARTITIONS
                        );
                        uint32_t global_idx1 = mpi::UniformGridRow::get_vertex_index(
                            {coords1[0], coords1[1], coords1[2]}, min, max, GLOBAL_PARTITIONS
                        );

                        if (global_idx0 != global_idx1) {
                            mesh.data(vh0).Collasable = false;
                            mesh.data(vh1).Collasable = false;
                        } else {
                            uint32_t idx0 = mesh.data(vh0).NodeIdx;
                            uint32_t idx1 = mesh.data(vh1).NodeIdx;

                            if (idx0 != idx1) {
                                mesh.data(vh0).Collasable = false;
                                mesh.data(vh1).Collasable = false;
                            } 
                        }

                    }

                    #pragma omp for schedule(static)
                    for (size_t i = 0; i < mesh.n_edges(); ++i) {
                        auto eh = qems::QEMMesh::EdgeHandle(i);

                        if(uniform_grid.add_edge(mesh, eh)) {
                            auto heh = mesh.halfedge_handle(eh, 0);
                            auto vh0 = mesh.from_vertex_handle(heh);
                            auto vh1 = mesh.to_vertex_handle(heh);

                            Eigen::Matrix4d Q = mesh.data(vh0).Quadric + mesh.data(vh1).Quadric;
                            Eigen::Vector4d newV = qems::compute_new_best_vertex(mesh, eh, Q);

                            mesh.data(eh).Error = newV.transpose() * Q * newV;
                            mesh.data(eh).NewVertex = newV;
                        }
                    }

                    #pragma omp for schedule(static)
                    for(size_t i = 0; i < mesh.n_faces(); i++) {
                        auto fh = qems::QEMMesh::FaceHandle(i);
                        uniform_grid.increment_collasable_faces(mesh, fh);
                    }
                }
                #pragma omp parallel for schedule(dynamic, 1)
                for (const auto &cell : uniform_grid) {
                    auto pq = qems::QEMPriorityQueue(qems::QEMEdgeCompare(mesh), cell.edges);

                    uint32_t local_num_faces = cell.collasable_faces; 
                    float total_faces = static_cast<float>(uniform_grid.total_collasable_faces());
                    float cell_faces  = static_cast<float>(local_num_faces);

                    float fraction = (total_faces > 0.0) ? (cell_faces / total_faces) : 0.0;
                    float target_d = static_cast<float>(TARGET_FACES) * fraction;

                    uint32_t local_target = static_cast<uint32_t>(std::floor(target_d));

                    qems::simplification(mesh, local_target, local_num_faces, pq);
                }

                mesh.garbage_collection();
                subdivision = next_step(subdivision);
            }
        }
        PROFILING_PRINT();

        qems::mesh_to_row_data(mesh, vertices, faces, idx_mapping);
        mpi::sync_send(0, msg);
    }
}
