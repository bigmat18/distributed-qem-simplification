#pragma once

#include <mesh_import.hpp>
#include <async_send.hpp>
#include <qem_mesh.hpp>
#include <qem_simp.hpp>
#include <sync_send_recv.hpp>
#include <message_layout.hpp>
#include <packed_message.hpp>
#include <uniform_grid.hpp>
#include <mpmc_queue.hpp>

#define CSTM_TAG_BB 1
#define CSTM_TAG_VERT 2
#define CSTM_TAG_FACE 3
#define CSTM_TAG_NAME 4
#define CSTM_MESH 5

inline auto get_layout() {
    mpi::MessageLayout layout(CSTM_MESH);
    layout 
     .add_buffer<char>(CSTM_TAG_NAME)
     .add_buffer<double, 6>(CSTM_TAG_BB)
     .add_buffer<float>(CSTM_TAG_VERT)
     .add_buffer<uint32_t>(CSTM_TAG_FACE);

    return layout;
}

inline int next_step(int n) {
    if (n == 1) return 0;
    return (n % 2 != 0) ? (n - 1) / 2 : (n > 4 ? n / 2 + 1 : 1);
}
