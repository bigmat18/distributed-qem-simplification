#pragma once

#include <cstdint>
#include <type_traits>
#include <variant>
#include <vector>

#include <utils.hpp>
#include <mpi.h>

#include "logging.hpp"
#include "message_layout.hpp"
#include "packed_message.hpp"

namespace mpi {

class AsyncSend {
    // List of buffers:
    // - [(key1, [type1]), (key2, [type2]) ...] buffer 1
    // - [(key1, [type1]), (key2, [type2]) ...] buffer 2
    // ...
    std::vector<PackedMessage> messages_;
    std::vector<MPI_Request> requests_;
    bool waited = false;

    uint32_t num_data_buffers_;
    uint32_t num_swap_buffers_;
    uint32_t active_buffer_idx_ = 0;
public:
    AsyncSend(MessageLayout layout, uint32_t num_buffers = 2) {
        num_swap_buffers_ = num_buffers;
        num_data_buffers_ = static_cast<uint32_t>(layout.size());
        messages_.insert(messages_.end(), num_buffers, {layout});
        requests_ = std::vector<MPI_Request>(
            num_swap_buffers_ * num_data_buffers_, 
            MPI_REQUEST_NULL
        );
    }

    ~AsyncSend() { 
        if (!requests_.empty()) {
            MPI_Waitall(static_cast<int>(requests_.size()), 
                        requests_.data(), 
                        MPI_STATUS_IGNORE);
        }
    }

    void isend(const int dest) {
        massert(waited, "First call wait()");
        const uint32_t index = active_buffer_idx_ * num_data_buffers_; 
        const auto& message = messages_[active_buffer_idx_];

        uint32_t buffer_counter = 0;
        for (const auto& [key, value] : message) {
            const auto& [buffer, is_static] = value; 

            std::visit([&](auto &buf) {
                using VecT = std::decay_t<decltype(buf)>;
                using T    = typename VecT::value_type;

                MPI_Request& request = requests_[index + buffer_counter];
                MPI_Datatype type = PackedMessage::get_mpi_type<T>(); 

                MPI_Isend(buf.data(), buf.size(),
                          type, dest, key, MPI_COMM_WORLD, &request);
            }, buffer);
            buffer_counter++;
        }
        waited = false;
    }

    inline PackedMessage& get_message() { 
        return messages_[active_buffer_idx_]; 
    }

    inline const PackedMessage& get_message() const { 
        return messages_[active_buffer_idx_]; 
    }

    inline void wait() {
        if (requests_.empty()) 
            return;

        int flag = 0;
        uint32_t search_idx = active_buffer_idx_;

        while (true) {
            search_idx = (search_idx + 1) % num_swap_buffers_;

            uint32_t index = search_idx * num_data_buffers_;
            MPI_Testall(num_data_buffers_, 
                        &requests_[index], 
                        &flag, 
                        MPI_STATUSES_IGNORE);

            if (flag) {
                active_buffer_idx_ = search_idx;
                break;
            }
        }
        waited = true;
    }
};

}



