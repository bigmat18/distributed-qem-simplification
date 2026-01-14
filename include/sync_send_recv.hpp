#pragma once

#include "message_layout.hpp"
#include "packed_message.hpp"
#include <mpi.h>

namespace mpi {

template<bool packed>
void __sync_send(const int dest, PackedMessage &message);

template<bool packed>
int __sync_recv(PackedMessage& message, int source);


// ==============================================
// =================== PACKED ===================
// ==============================================
template<>
inline void __sync_send<true>(const int dest, PackedMessage& message) {
    const auto tag = message.tag();
    const auto& packed_message = message.pack_data();

    MPI_Send(packed_message.data(), packed_message.size(), 
             MPI_PACKED, dest, tag, MPI_COMM_WORLD);
}

template<>
inline int __sync_recv<true>(PackedMessage& message, int source) {
    const auto tag = message.tag();
    MPI_Status status;
    int count;

    MPI_Probe(source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    const auto arrival_tag = status.MPI_TAG;

    if (arrival_tag == CSTM_TAG_END || arrival_tag != tag)
         return -1;

    if (source == MPI_ANY_SOURCE)
        source = status.MPI_SOURCE;

    MPI_Get_count(&status, MPI_PACKED, &count);
    std::vector<char> packed_data;
    packed_data.resize(count);
    MPI_Recv(packed_data.data(), count, MPI_PACKED, 
             source, tag, MPI_COMM_WORLD, &status);
    message.unpack_data(std::move(packed_data));

    return source;
}

// ================================================
// =================== UNPACKED =================== 
// ================================================
template<>
inline void __sync_send<false>(const int dest, PackedMessage& message) {
    if (message.tag() == CSTM_TAG_END) {
        MPI_Send(nullptr, 0, MPI_BYTE, dest, 
                 CSTM_TAG_END, MPI_COMM_WORLD);
        return;
    }

    for (auto &[key, value] : message.data_) {
        std::visit([&](auto &buffer) {
            using VecT = std::decay_t<decltype(buffer)>;
            using T    = typename VecT::value_type;

            MPI_Datatype type = PackedMessage::get_mpi_type<T>(); 

            auto& send_buffer = message.get_buffer<T>(key);
            send_buffer = std::move(buffer);
            MPI_Send(send_buffer.data(), send_buffer.size(), 
                     type, dest, key, MPI_COMM_WORLD);
        }, value);
    }
}

template<>
inline int __sync_recv<false>(PackedMessage& message, int source) {
    MPI_Status status;
    int count;
    bool end = false;

    for (auto &[key, value] : message.data_) {

        std::visit([&](auto &buffer) {
            using VecT = std::decay_t<decltype(buffer)>;
            using T    = typename VecT::value_type;
            MPI_Datatype type = PackedMessage::get_mpi_type<T>(); 
            auto& recv_buffer = message.get_buffer<T>(key);

            MPI_Probe(source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            const auto arrival_tag = status.MPI_TAG;

            if (arrival_tag == CSTM_TAG_END || arrival_tag != key) {
                end = true;
                return;
            }

            if (source == MPI_ANY_SOURCE)
                source = status.MPI_SOURCE;

            MPI_Get_count(&status, MPI_PACKED, &count);

            if (recv_buffer.size() != count)
                recv_buffer.resize(count); 

            MPI_Recv(recv_buffer.data(), count, type, 
                     source, key, MPI_COMM_WORLD, &status);
        }, value);

        if (end) return -1;
    }
    return source;
}



inline void packed_sync_send(const int dest, PackedMessage& message) 
    {__sync_send<true>(dest, message);}

inline int packed_sync_recv(PackedMessage& message, int source = MPI_ANY_SOURCE) 
    {return __sync_recv<true>(message, source);}

inline void unpacked_sync_send(const int dest, PackedMessage& message) 
    {__sync_send<false>(dest, message);}

inline int unpacked_sync_recv(PackedMessage& message, int source = MPI_ANY_SOURCE) 
    {return __sync_recv<false>(message, source);}

}
