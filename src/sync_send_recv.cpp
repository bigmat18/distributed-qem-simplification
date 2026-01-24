#include "massert.hpp"
#include <mpi.h>
#include <sync_send_recv.hpp>

namespace mpi {

void sync_send(const int dest, const PackedMessage& message) {
    if (message.tag() == CSTM_TAG_END) {
        MPI_Send(nullptr, 0, MPI_BYTE, dest, 
                 CSTM_TAG_END, MPI_COMM_WORLD);
        return;
    }


    std::vector<char> packed_message;
    message.pack_data(packed_message);
    MPI_Send(packed_message.data(), packed_message.size(), MPI_PACKED, 
             dest, message.tag(), MPI_COMM_WORLD);

    for (const auto& [key, value] : message.buffer_data_) {
        const auto& [buffer, is_static] = value;

        std::visit([&](auto &buf) {
            using VecT = std::decay_t<decltype(buf)>;
            using T    = typename VecT::value_type;

            MPI_Datatype type = PackedMessage::get_mpi_type<T>(); 
            MPI_Send(buf.data(), buf.size(), 
                     type, dest, key, MPI_COMM_WORLD);
        }, buffer);
    }
}

int sync_recv(PackedMessage& message, int source) {
    MPI_Status status;
    int count;

    MPI_Probe(source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    if (status.MPI_TAG == CSTM_TAG_END)
        return -1;

    massert(status.MPI_TAG == message.tag(), "Error in key");

    if (source == MPI_ANY_SOURCE)
        source = status.MPI_SOURCE;

    MPI_Get_count(&status, MPI_PACKED, &count);
    std::vector<char> packed_message;
    packed_message.resize(count); 

    MPI_Recv(packed_message.data(), count, MPI_PACKED, 
             source, message.tag(), MPI_COMM_WORLD, &status);
    message.unpack_data(packed_message);

    bool end = false;
    for (auto& [key, value] : message.buffer_data_) {
        auto& [buffer, is_static] = value;

        std::visit([&](auto& buf) {
            using VecT = std::decay_t<decltype(buf)>;
            using T    = typename VecT::value_type;

            MPI_Datatype type = PackedMessage::get_mpi_type<T>(); 

            if (!is_static) {
                MPI_Probe(source, key, MPI_COMM_WORLD, &status);
                MPI_Get_count(&status, type, &count);

                if (buf.size() != count)
                    buf.resize(count); 

                MPI_Recv(buf.data(), count, type, 
                         source, key, MPI_COMM_WORLD, &status);
            } else {
                MPI_Recv(buf.data(), buf.size(), type, 
                         source, key, MPI_COMM_WORLD, &status);
            }

        }, buffer);
    }

    return source;
}

}
