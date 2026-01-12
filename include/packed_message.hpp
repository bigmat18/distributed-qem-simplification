#pragma once 

#include "massert.hpp"
#include "message_layout.hpp"
#include <cstdint>
#include <variant>
#include <vector>
#include <map>

#include <utils.hpp>
#include <mpi.h>

namespace mpi {

class PackedMessage {
    using MPI_CUSTOM_TAG = int;
    using DataBlock = std::map<
        MPI_CUSTOM_TAG, 
        std::variant<
            std::vector<char>, std::vector<int>,
            std::vector<uint32_t>, std::vector<uint64_t>,
            std::vector<float>, std::vector<double>
        >
    >;

    DataBlock data_;
    std::vector<char> packed_data_;
    MPI_CUSTOM_TAG tag_ = CSTM_TAG_END;

public:
    PackedMessage() = default;

    PackedMessage(const MessageLayout layout) : 
        tag_(layout.tag()) 
    {
        for (const auto [key, value] : layout) {
            std::visit([&](auto& buffer) {
                using Holder = std::decay_t<decltype(buffer)>;
                using T      = typename Holder::type;
                data_.try_emplace(key, std::vector<T>{});
            }, value);
        }
    }

    template<typename T> requires MessageSupportedTypes<T>
    std::vector<T>& get_buffer(const MPI_CUSTOM_TAG tag) {
        using VecT = std::vector<T>;
        
        auto it = data_.find(tag);
        massert(it != data_.end(), "Tag " + std::to_string(tag) + " not found in layout.");

        auto *vec = std::get_if<VecT>(&(it->second));
        massert(vec != nullptr, "Type mismatch for tag " + std::to_string(tag));
        
        return *vec;
    }

    inline MPI_CUSTOM_TAG tag() const { return tag_; }

private:

    const std::vector<char>& pack_data() {
        packed_data_.clear();
        std::size_t total_size = 0;

        for (const auto &[key, value] : data_) {
            std::visit([&](auto& buffer) {
                using VecT = std::decay_t<decltype(buffer)>;
                using T    = typename VecT::value_type;
                MPI_Datatype mpi_type = get_mpi_type<T>();

                int size;
                MPI_Pack_size(1, MPI_UINT32_T, MPI_COMM_WORLD, &size);
                total_size += size;

                MPI_Pack_size(static_cast<int>(buffer.size()), 
                              mpi_type, MPI_COMM_WORLD, &size);
                total_size += size;


            }, value);
        }
        packed_data_.resize(total_size);

        int position = 0;
        for (auto &[key, value] : data_) {
            std::visit([&](auto& buffer) {
                using VecT = std::decay_t<decltype(buffer)>;
                using T    = typename VecT::value_type;
                MPI_Datatype mpi_type = get_mpi_type<T>();

                uint32_t size = static_cast<uint32_t>(buffer.size());
                MPI_Pack(&size, 1, MPI_UINT32_T, packed_data_.data(), 
                         total_size, &position, MPI_COMM_WORLD);

                if (!buffer.empty()) {
                    MPI_Pack(buffer.data(), buffer.size(), mpi_type,
                             packed_data_.data(), total_size, &position, MPI_COMM_WORLD);
                }

                buffer.clear();
                buffer.shrink_to_fit();
            }, value);
        }
        packed_data_.resize(position);
        return packed_data_;
    } 

    void unpack_data(std::vector<char> packed_data) {
        int position = 0;
        uint32_t total_size = static_cast<uint32_t>(packed_data.size());

        for (auto& [key, value] : data_) {
            uint32_t size = 0;
            MPI_Unpack(packed_data.data(), total_size, &position, 
                       &size, 1, MPI_UINT32_T, MPI_COMM_WORLD); 

            std::visit([&](auto& buffer) {
                using VecT = std::decay_t<decltype(buffer)>;
                using T    = typename VecT::value_type;
                MPI_Datatype mpi_type = get_mpi_type<T>();

                buffer.resize(size);

                MPI_Unpack(packed_data.data(), total_size, &position, 
                           buffer.data(), size, mpi_type, MPI_COMM_WORLD);

            }, value);
        }
        packed_data.clear();
    }

    template<typename T>
    constexpr MPI_Datatype get_mpi_type() noexcept {
        if constexpr (std::is_same_v<T, int>)           return MPI_INT;
        else if constexpr (std::is_same_v<T, double>)   return MPI_DOUBLE;
        else if constexpr (std::is_same_v<T, uint32_t>) return MPI_UINT32_T;
        else if constexpr (std::is_same_v<T, uint64_t>) return MPI_UINT64_T;
        else if constexpr (std::is_same_v<T, float>)    return MPI_FLOAT;
        else if constexpr (std::is_same_v<T, char>)     return MPI_CHAR;
        else {
            static_assert(
                std::is_same_v<T, void>,
                "This is not a valid type for MPI_Isend"
            );
        }
    }

    friend class AsyncSend;

    friend void sync_send(const int dest, PackedMessage& message);  
    friend int sync_recv(PackedMessage& message, int source);
};

}
