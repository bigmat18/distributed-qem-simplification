#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <utils.hpp>
#include <mpi.h>

#include "message_layout.hpp"
#include "packed_message.hpp"

namespace mpi {

class AsyncSend {
    // List of buffers:
    // - [(key1, [type1]), (key2, [type2]) ...] buffer 1
    // - [(key1, [type1]), (key2, [type2]) ...] buffer 2
    std::vector<PackedMessage> messages_;
    std::vector<MPI_Request> requests_;

    uint32_t num_swap_buffers_ = 2;
    uint32_t active_buffer_idx_ = 0;

public:
    AsyncSend(MessageLayout layout, uint32_t num_buffers = 2) :   
        num_swap_buffers_(num_buffers),  
        requests_(num_buffers, MPI_REQUEST_NULL)  
    { 
        PackedMessage msg(layout);
        messages_.insert(messages_.end(), num_buffers, msg);
    }

    ~AsyncSend() { wait(); }

    void isend(const int dest, PackedMessage::DataBlock data) {
        uint32_t index = active_buffer_idx_;
        MPI_Waitall(1, &requests_[index], MPI_STATUS_IGNORE);

        auto &message = messages_[index];
        for (auto &[key, value] : data) {
            std::visit(
                [&](auto &buffer) {
                    using VecT = std::decay_t<decltype(buffer)>;
                    using T    = typename VecT::value_type;
                    message.get_buffer<T>(key) = std::move(buffer);
                }, 
                value  
            );
        }
        const auto& packed_data = message.pack_data();
        const auto tag = message.tag();

        MPI_Request& request = requests_[active_buffer_idx_];
        MPI_Isend(packed_data.data(), packed_data.size(), 
                  MPI_PACKED, dest, tag, MPI_COMM_WORLD, &request);

        active_buffer_idx_ = (active_buffer_idx_ + 1) % num_swap_buffers_;
    }

    inline void wait() {
        if (!requests_.empty()) {
            MPI_Waitall(static_cast<int>(requests_.size()), 
                        requests_.data(), 
                        MPI_STATUS_IGNORE);
        }
    }
};

}



