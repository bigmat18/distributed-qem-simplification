#pragma once
#include "massert.hpp"
#include <cmath>
#include <cstdint>
#include <qem_mesh.hpp>
#include <qem_simp.hpp>
#include <qem_mesh.hpp>
#include <uniform_grid_row.hpp>
#include <mesh_import.hpp>
#include <message_layout.hpp>
#include <packed_message.hpp>
#include <async_send.hpp>
#include <sync_send_recv.hpp>
#include <mpmc_queue.hpp>

#define CSTM_TAG_BB 1
#define CSTM_TAG_VERT 2
#define CSTM_TAG_FACE 3
#define CSTM_TAG_NAME 4

#define CSTM_TAG_CELL_ID 5
#define CSTM_TAG_CELL_PART_LVL 6
#define CSTM_TAG_IDX_MAP 7
#define CSTM_TAG_FINAL_TARGET 8

#define CSTM_MESH 9

inline auto get_layout() {
    mpi::MessageLayout layout(CSTM_MESH);
    layout
     .add_element<uint32_t, 3>(CSTM_TAG_CELL_ID)
     .add_element<uint32_t, 2>(CSTM_TAG_CELL_PART_LVL)
     .add_element<uint32_t, 1>(CSTM_TAG_FINAL_TARGET)
     .add_element<double, 6>(CSTM_TAG_BB)
     .add_element<char>(CSTM_TAG_NAME)
     .add_buffer<float>(CSTM_TAG_VERT)
     .add_buffer<uint32_t>(CSTM_TAG_FACE)
     .add_buffer<uint32_t>(CSTM_TAG_IDX_MAP);
    return layout;
}

inline int next_step(int n) {
    if (n == 1) return 0;
    return (n % 2 != 0) ? (n - 1) / 2 : (n > 4 ? n / 2 + 1 : 1);
}

inline int get_dest(uint32_t idx, uint32_t split, uint32_t num_worker, uint32_t shift = 0) {
    massert(num_worker >= 1, "Num worker must be >= 1");

    auto coords = mpi::UniformGridRow::get_cell_indices(idx, split);
    uint32_t partitions = static_cast<uint32_t>(std::floor(std::cbrt(num_worker)));     
    uint32_t block_dim = std::ceil(static_cast<float>(split) / static_cast<float>(partitions));

    if (block_dim == 0) block_dim = 1;

    uint32_t bx = std::min(coords.x() / block_dim, partitions - 1);
    uint32_t by = std::min(coords.y() / block_dim, partitions - 1);
    uint32_t bz = std::min(coords.z() / block_dim, partitions - 1);
 
    uint32_t total_partitions = partitions * partitions * partitions;
    uint32_t partition_id = (bz * partitions * partitions) + 
                            (by * partitions) + bx;

    if (num_worker <= total_partitions) {
        return partition_id % num_worker;
    }

    uint32_t K = (num_worker - 1 - partition_id) / total_partitions + 1;
    massert(K != 0, "No worker for partition " + std::to_string(partition_id));

    uint32_t local_x = coords.x() % block_dim;
    uint32_t local_y = coords.y() % block_dim;
    uint32_t local_z = coords.z() % block_dim;

    uint32_t block_volume = block_dim * block_dim * block_dim;
    uint32_t local_index = (local_z * block_dim * block_dim) + (local_y * block_dim) + local_x;
    
    uint32_t rank = (local_index * K) / block_volume;
    if (rank >= K) rank = K - 1;
    return ((partition_id + (rank * total_partitions)) + shift) % num_worker;
}


struct PackedMessageCompare {
    bool operator()(const mpi::PackedMessage& msg1,
                    const mpi::PackedMessage& msg2) const 
    {
        uint32_t file_id1 = msg1.get_element<uint32_t>(CSTM_TAG_CELL_ID)[2];
        uint32_t file_id2 = msg2.get_element<uint32_t>(CSTM_TAG_CELL_ID)[2];
        if (file_id1 == file_id2) {
            uint32_t part1 = msg1.get_element<uint32_t>(CSTM_TAG_CELL_PART_LVL)[1];
            uint32_t part2 = msg2.get_element<uint32_t>(CSTM_TAG_CELL_PART_LVL)[1];
            return part1 < part2;
        }
        return file_id1 > file_id2;
    }
};

//MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
//if (flag) {
    //mpi::PackedMessage tmp_msg(layout);
    //mpi::sync_recv(tmp_msg, status.MPI_SOURCE);
    //tasks.push(tmp_msg);
//}

//if (!tasks.empty()) {
    //if(cmp(msg, tasks.front())) { // msg < task ==> compute task
        //std::pop_heap(tasks.begin(), tasks.end(), cmp);
        //auto task = std::move(tasks.back());
        //tasks.pop_back();
        //tasks.push_back(std::move(msg));
        //std::push_heap(tasks.begin(), tasks.end(), cmp);
        //msg = std::move(task);

        //id = msg.get_element<uint32_t>(CSTM_TAG_CELL_ID);
        //part_lvl = msg.get_element<uint32_t>(CSTM_TAG_CELL_PART_LVL);
        //final_target = msg.get_element<uint32_t>(CSTM_TAG_FINAL_TARGET);
        //bb = msg.get_element<double>(CSTM_TAG_BB);
        //name = msg.get_element<char>(CSTM_TAG_NAME);

        //vertices = msg.get_buffer<float>(CSTM_TAG_VERT);
        //faces = msg.get_buffer<uint32_t>(CSTM_TAG_FACE);
        //idx_mapping = msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP);
    //}
//}

