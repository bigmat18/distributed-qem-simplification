#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <cxxopts.hpp>

#include <mpi.h>
#include <utils.hpp>

#include "utils.hpp"

inline void Main_Master_V1(int pid, int num_procs,
                        const std::string INPUT,
                        const uint32_t START_PARTITIONS,
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
        metadata.max_coords, START_PARTITIONS 
    );

    uint32_t cell_id = 0;
    float total_faces = static_cast<float>(faces.size()/3);
    uint32_t final_target = static_cast<uint32_t>(std::floor(total_faces * TARGET));

    for (auto &cell : uniform_grid) {
        async_sender.wait();
        auto& msg = async_sender.get_message();

        msg.get_element<uint32_t>(CSTM_TAG_CELL_PART_LVL) = {START_PARTITIONS, START_PARTITIONS};
        msg.get_element<uint32_t>(CSTM_TAG_FINAL_TARGET) = { final_target };

        auto& bb = msg.get_element<double>(CSTM_TAG_BB);
        bb[0] = min.x(); bb[1] = min.y(); bb[2] = min.z();
        bb[3] = max.x(); bb[4] = max.y(); bb[5] = max.z();

        msg.get_element<char>(CSTM_TAG_NAME).assign(metadata.name.begin(), metadata.name.end());
        msg.get_element<uint32_t>(CSTM_TAG_CELL_ID) = {cell_id, cell_id, 0};

        msg.get_buffer<float>(CSTM_TAG_VERT) = std::move(cell.vertices);
        msg.get_buffer<uint32_t>(CSTM_TAG_FACE) = std::move(cell.faces);
        msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP) = std::move(cell.indices_mapping);

        uint32_t dest = get_dest(cell_id, START_PARTITIONS, num_procs-1, 0);
        async_sender.isend(dest+1);
        cell_id++;
    }
   
    mpi::PackedMessage recv_msg(layout);
    mpi::sync_recv(recv_msg);
    const auto& recv_name = recv_msg.get_element<char>(CSTM_TAG_NAME);
    const auto& recv_vertices = recv_msg.get_buffer<float>(CSTM_TAG_VERT);
    const auto& recv_faces = recv_msg.get_buffer<uint32_t>(CSTM_TAG_FACE);

    if (!recv_vertices.empty()) {
        std::string str_name(recv_name.data(), recv_name.size());
        qems::export_mesh("out/" + str_name, recv_vertices, recv_faces);
    }

    mpi::PackedMessage final_msg;
    for (int w = 1; w < num_procs; ++w)
        mpi::sync_send(w, final_msg);
}
