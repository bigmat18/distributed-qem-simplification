#pragma once

#include "massert.hpp"
#include <cstdint>
#include <mpi.h>
#include <map>
#include <type_traits>
#include <variant>
#include <vector>
#include <utils.hpp>

namespace mpi {

template <typename T>
concept AsyncSendSupported =
    std::disjunction_v<
        std::is_same<T, char>,
        std::is_same<T, int>,
        std::is_same<T, uint32_t>,
        std::is_same<T, uint64_t>,
        std::is_same<T, float>,
        std::is_same<T, double>>;

class AsyncSend {
    using MPI_CUSTOM_TAG = int;
    using DataBlock = std::map<
        MPI_CUSTOM_TAG, 
        std::variant<
            std::vector<char>, std::vector<int>,
            std::vector<uint32_t>, std::vector<uint64_t>,
            std::vector<float>, std::vector<double>
        >
    >;

    // List of buffers:
    // - [(key1, [type1]), (key2, [type2]) ...] buffer 1
    // - [(key1, [type1]), (key2, [type2]) ...] buffer 2
    // ...
    std::vector<DataBlock> buffers_;
    std::vector<MPI_Request> requests_;

    uint32_t num_swap_buffers_ = 2;
    uint32_t active_buffer_idx_ = 0;

public:
    AsyncSend(uint32_t num_buffers = 2) : 
        num_swap_buffers_(num_buffers) 
    {
        for (uint32_t i = 0; i < num_swap_buffers_; i++)
            buffers_.push_back({});
    }

    template<typename T> requires AsyncSendSupported<T>
    AsyncSend& add_buffer(const MPI_CUSTOM_TAG tag) {
        for (uint32_t i = 0; i < num_swap_buffers_; ++i) {
            buffers_[i].try_emplace(tag, std::vector<T>{});
        }
        requests_.insert(requests_.end(), num_swap_buffers_, MPI_REQUEST_NULL);
        return *this;
    }
      
    void isend(const int dest, DataBlock&& data) {
        uint32_t num_buffers = buffers_[0].size();
        uint32_t index = active_buffer_idx_ * num_buffers;
        MPI_Waitall(num_buffers, &requests_[index], MPI_STATUS_IGNORE);
       
        uint32_t entry_idx = 0;
        for (auto &[tag, stored_variant] : buffers_[active_buffer_idx_]) {
            auto it = data.find(tag);
            if (it == data.end()) massert(false, "Missing tag in input data");
            auto &new_data_variant = it->second;

            std::visit(
                [&](auto &&vec_in) {
                    using VecT = std::decay_t<decltype(vec_in)>;
                    using T    = typename VecT::value_type;

                    update_buffer<T>(tag, std::move(vec_in));
                    auto &final_vec = std::get<VecT>(stored_variant);

                    if (final_vec.empty())
                        return;

                    MPI_Datatype mpi_type = get_mpi_type<T>();
                    MPI_Request &req = requests_[index + entry_idx];

                    MPI_Isend(final_vec.data(), static_cast<int>(final_vec.size()),
                              mpi_type, dest, tag, MPI_COMM_WORLD, &req);
                },
                new_data_variant
            );

            ++entry_idx;
        }
        active_buffer_idx_ = (active_buffer_idx_ + 1) % num_swap_buffers_;
    }

    inline void wait() {
        if (!requests_.empty()) {
            MPI_Waitall(static_cast<int>(requests_.size()), 
                        requests_.data(), 
                        MPI_STATUS_IGNORE);
        }
    }

private:
    template<typename T> requires AsyncSendSupported<T>
    void update_buffer(const MPI_CUSTOM_TAG tag, std::vector<T>&& data) {
        using VecT = std::vector<T>;
        auto &slot = buffers_[active_buffer_idx_][tag];

        if (auto *vec = std::get_if<VecT>(&slot))
            *vec = std::move(data);
        else
            massert(false, "Error for type " + std::string(typeid(T).name()));
    }

    template<typename T>
    constexpr MPI_Datatype get_mpi_type() noexcept {
        if constexpr (std::is_same_v<T, int>)           return MPI_INT;
        else if constexpr (std::is_same_v<T, double>)   return MPI_DOUBLE;
        else if constexpr (std::is_same_v<T, uint32_t>) return MPI_UNSIGNED;
        else if constexpr (std::is_same_v<T, uint64_t>) return MPI_UNSIGNED_LONG_LONG;
        else if constexpr (std::is_same_v<T, float>)    return MPI_FLOAT;
        else if constexpr (std::is_same_v<T, char>)     return MPI_CHAR;
        else {
            static_assert(
                std::is_same_v<T, void>,
                "This is not a valid type for MPI_Isend"
            );
        }
    }
};

}



