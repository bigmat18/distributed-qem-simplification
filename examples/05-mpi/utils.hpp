#pragma once
#include <qem_mesh.hpp>
#include <qem_simp.hpp>
#include <qem_mesh.hpp>
#include <ug_row_data.hpp>
#include <mesh_import.hpp>
#include <message_layout.hpp>
#include <packed_message.hpp>
#include <async_send.hpp>
#include <sync_send_recv.hpp>

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
     .add_element<uint32_t, 2>(CSTM_TAG_CELL_ID)
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
