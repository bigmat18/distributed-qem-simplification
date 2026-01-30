#include <cmath>
#include <cstdint>
#include <omp.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <cxxopts.hpp>

#include <mpi.h>
#include <utils.hpp>

#include "utils.hpp"

inline void Main_Master_V1(int pid, int num_procs,
                        const std::string INPUT,
                        const uint32_t GLOBAL_PARTITIONS,
                        uint32_t NUM_MESHES,
                        const float TARGET)
{
    mpi::MessageLayout layout = get_layout();
    mpi::AsyncSend async_sender(layout, (num_procs-1)*20);

    qems::MeshMetaData metadata;
    std::vector<float> vertices;
    std::vector<uint32_t> faces;
    const auto& min = metadata.min_coords;
    const auto& max = metadata.max_coords;

    qems::import_mesh(INPUT, metadata, vertices, faces);
    auto uniform_grid = mpi::UniformGridRow(
        vertices, faces, metadata.min_coords, 
        metadata.max_coords, GLOBAL_PARTITIONS 
    );

    uint32_t final_target = static_cast<uint32_t>(static_cast<float>(faces.size()/3) * TARGET);
    auto& cells = uniform_grid.cells();
    uint32_t cell_id = 0;
    for (int dest = 1; dest < num_procs; ++dest) {
        if (cell_id >= cells.size())
            continue;

        async_sender.wait();
        auto& cell = cells[cell_id];
        auto& msg = async_sender.get_message();

        auto& bb = msg.get_element<double>(CSTM_TAG_BB);
        bb[0] = min.x(); bb[1] = min.y(); bb[2] = min.z();
        bb[3] = max.x(); bb[4] = max.y(); bb[5] = max.z();

        msg.get_element<char>(CSTM_TAG_NAME).assign(metadata.name.begin(), metadata.name.end());
        msg.get_element<uint32_t>(CSTM_TAG_CELL_ID) = {cell_id};
        msg.get_element<uint32_t>(CSTM_TAG_FINAL_TARGET) = {final_target};
        msg.get_buffer<float>(CSTM_TAG_VERT) = std::move(cell.vertices);
        msg.get_buffer<uint32_t>(CSTM_TAG_FACE) = std::move(cell.faces);
        msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP) = std::move(cell.indices_mapping);

        async_sender.isend(dest);
        cell_id++;
    }

    mpi::PackedMessage recv_msg(layout);
    std::vector<mpi::PackedMessage> mesh_cells;
    while (mesh_cells.size() < cells.size()) {
        const int dest = mpi::sync_recv(recv_msg);
        const auto& recv_name = recv_msg.get_element<char>(CSTM_TAG_NAME);
        std::string str_name(recv_name.data(), recv_name.size());
        mesh_cells.push_back(recv_msg);

        if (cell_id >= cells.size())
            continue;

        async_sender.wait();
        auto& cell = cells[cell_id];
        auto& msg = async_sender.get_message();

        auto& bb = msg.get_element<double>(CSTM_TAG_BB);
        bb[0] = min.x(); bb[1] = min.y(); bb[2] = min.z();
        bb[3] = max.x(); bb[4] = max.y(); bb[5] = max.z();

        msg.get_element<char>(CSTM_TAG_NAME).assign(metadata.name.begin(), metadata.name.end());
        msg.get_element<uint32_t>(CSTM_TAG_CELL_ID) = {cell_id};
        msg.get_element<uint32_t>(CSTM_TAG_FINAL_TARGET) = {final_target};
        msg.get_buffer<float>(CSTM_TAG_VERT) = std::move(cell.vertices);
        msg.get_buffer<uint32_t>(CSTM_TAG_FACE) = std::move(cell.faces);
        msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP) = std::move(cell.indices_mapping);

        async_sender.isend(dest);
        cell_id++;
    }

    {
        PROFILING_SCOPE("PID:"+ std::to_string(pid) +", merging cells");
        auto& name = mesh_cells[0].get_element<char>(CSTM_TAG_NAME);
        std::string str_name(name.data(), name.size());
        auto& bb = mesh_cells[0].get_element<double>(CSTM_TAG_BB);
        Eigen::Vector3d min, max;
        min.x() = bb[0]; min.y() = bb[1]; min.z() = bb[2];
        max.x() = bb[3]; max.y() = bb[4]; max.z() = bb[5];

        mpi::UniformGridRow uniform_grid({}, {}, {}, min, max, GLOBAL_PARTITIONS);  
        for(auto& msg : mesh_cells) {
            uint32_t id = msg.get_element<uint32_t>(CSTM_TAG_CELL_ID)[0];
            auto& cell = uniform_grid.cells()[id];
            cell.vertices = std::move(msg.get_buffer<float>(CSTM_TAG_VERT));
            cell.faces = std::move(msg.get_buffer<uint32_t>(CSTM_TAG_FACE));
            cell.indices_mapping = std::move(msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP));
        }
        std::vector<float> out_vertices;
        std::vector<uint32_t> out_faces;
        std::vector<uint32_t> out_idx_map;
        uniform_grid.merge_cells(out_vertices, out_faces, out_idx_map);

        qems::QEMMesh mesh;
        mesh.request_vertex_status();
        mesh.request_edge_status();
        mesh.request_face_status();
        mesh.request_halfedge_status();
        mesh.request_vertex_normals();
        mesh.request_face_normals();

        qems::row_data_to_mesh(out_vertices, out_faces, out_idx_map, mesh);
        mesh.update_normals();
        std::vector<qems::QEMMesh::EdgeHandle> edges;

        for (auto vh : mesh.vertices()) {
            auto coords = mesh.point(vh);
            mesh.data(vh).Quadric = qems::compute_vertex_quadratic(mesh, vh);
        }


        for (auto eh : mesh.edges()) {
            auto heh = mesh.halfedge_handle(eh, 0);
            auto vh0 = mesh.from_vertex_handle(heh);
            auto vh1 = mesh.to_vertex_handle(heh);

            Eigen::Matrix4d Q = mesh.data(vh0).Quadric + mesh.data(vh1).Quadric;
            Eigen::Vector4d newV = qems::compute_new_best_vertex(mesh, eh, Q);

            mesh.data(eh).Error = newV.transpose() * Q * newV;
            mesh.data(eh).NewVertex = newV;
            edges.push_back(eh);
        }

        auto pq = qems::QEMPriorityQueue(qems::QEMEdgeCompare(mesh), edges);

        qems::simplification(mesh,final_target, mesh.n_faces(), pq);
        mesh.garbage_collection();
        qems::mesh_to_row_data(mesh, out_vertices, out_faces, out_idx_map);

        qems::export_mesh("out/"+str_name, out_vertices, out_faces);
    }
    PROFILING_PRINT();
    
    mpi::PackedMessage final_msg;
    for (int w = 1; w < num_procs; ++w)
        mpi::sync_send(w, final_msg);
}
