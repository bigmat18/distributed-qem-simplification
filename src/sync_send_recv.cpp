#include <sync_send_recv.hpp>

namespace mpi {

void sync_send(const int dest, const PackedMessage& message) {
    if (message.tag() == CSTM_TAG_END) {
        MPI_Send(nullptr, 0, MPI_BYTE, dest, 
                 CSTM_TAG_END, MPI_COMM_WORLD);
        return;
    }

    for (const auto& [key, value] : message) {
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
    bool end = false;

    for (auto& [key, value] : message) {
        auto& [buffer, is_static] = value;

        std::visit([&](auto& buf) {
            using VecT = std::decay_t<decltype(buf)>;
            using T    = typename VecT::value_type;

            MPI_Datatype type = PackedMessage::get_mpi_type<T>(); 

            if (!is_static) {
                MPI_Probe(source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if (status.MPI_TAG == CSTM_TAG_END || status.MPI_TAG != key) {
                    end = true;
                    return;
                }

                if (source == MPI_ANY_SOURCE)
                    source = status.MPI_SOURCE;

                MPI_Get_count(&status, type, &count);

                if (buf.size() != count)
                    buf.resize(count); 

                MPI_Recv(buf.data(), count, type, 
                         source, key, MPI_COMM_WORLD, &status);
            } else {
                MPI_Recv(buf.data(), buf.size(), type, 
                         source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                if (status.MPI_TAG == CSTM_TAG_END || status.MPI_TAG != key) {
                    end = true;
                    return;
                }

                if (source == MPI_ANY_SOURCE)
                    source = status.MPI_SOURCE;
            }

        }, buffer);

        if (end) 
            return -1;
    }
    return source;
}

}
