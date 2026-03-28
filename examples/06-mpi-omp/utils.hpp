#pragma once
#include <cmath>
#include <cstdint>
#include <qem_mesh.hpp>
#include <qem_simp.hpp>
#include <qem_mesh.hpp>
#include <mesh_import.hpp>
#include <message_layout.hpp>
#include <packed_message.hpp>
#include <async_send.hpp>
#include <sync_send_recv.hpp>
#include <mpmc_queue.hpp>
#include <uniform_grid_row.hpp>
#include <uniform_grid_qem.hpp>

#define CSTM_TAG_NAME 1
#define CSTM_TAG_CELL_ID 2
#define CSTM_TAG_VERT 3
#define CSTM_TAG_FACE 4
#define CSTM_TAG_IDX_MAP 5
#define CSTM_TAG_BB 6
#define CSTM_TAG_FINAL_TARGET 7

#define CSTM_MESH 8

inline auto get_layout() {
    mpi::MessageLayout layout(CSTM_MESH);
    layout
     .add_element<uint32_t, 1>(CSTM_TAG_CELL_ID)
     .add_element<double, 6>(CSTM_TAG_BB)
     .add_element<uint32_t, 1>(CSTM_TAG_FINAL_TARGET)
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


inline uint32_t next_pow(uint32_t n) {
    if (n == 0) return 1;
    uint32_t p = 1;
    while (p < n)
        p = p * 2;
    return p;
}

using BoundingBox = std::pair<Eigen::Vector3d, Eigen::Vector3d>;

inline BoundingBox get_cell_bb(const Eigen::Vector3d& min,
                               const Eigen::Vector3d& max,
                               uint32_t n,       
                               uint32_t idx) 
{
    assert(n > 0);
    Eigen::Vector3d size = max - min;
    Eigen::Vector3d cellSize = size / static_cast<double>(n);

    uint32_t n2 = n * n;
    uint32_t rem = idx % n2;

    uint32_t X = rem % n;
    uint32_t Y = rem / n;
    uint32_t Z = idx / n2;

    Eigen::Vector3d idx_3d(
        static_cast<double>(X),
        static_cast<double>(Y),
        static_cast<double>(Z)
    );

    Eigen::Vector3d localMin = min + idx_3d.cwiseProduct(cellSize);
    Eigen::Vector3d localMax = localMin + cellSize;

    return {localMin, localMax};
}
