#pragma once

#include "packed_message.hpp"
#include <mpi.h>

namespace mpi {
#define CSTM_TAG_END 0

inline void sync_send(const int dest, PackedMessage& message) {
    const auto tag = message.tag();
    const auto& packed_message = message.pack_data();

    MPI_Send(packed_message.data(), packed_message.size(), 
             MPI_PACKED, dest, tag, MPI_COMM_WORLD);
}

inline bool sync_recv(const int source, PackedMessage& message) {
    const auto tag = message.tag();
    MPI_Status status;
    int count;

    MPI_Probe(source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    const auto arrival_tag = status.MPI_TAG;

    if (arrival_tag == CSTM_TAG_END || arrival_tag != tag)
         return false;

    MPI_Get_count(&status, MPI_PACKED, &count);
    std::vector<char> packed_data(count);
    MPI_Recv(packed_data.data(), count, MPI_PACKED, source, tag, MPI_COMM_WORLD, &status);
    message.unpack_data(std::move(packed_data));

    return true;
}

}
