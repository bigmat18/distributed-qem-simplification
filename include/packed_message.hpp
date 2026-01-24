#pragma once 

#include <cstdint>
#include <variant>
#include <vector>
#include <map>

#include <utils.hpp>
#include <mpi.h>

#include "logging.hpp"
#include "massert.hpp"
#include "message_layout.hpp"

namespace mpi {

class PackedMessage {
    using MPI_CUSTOM_TAG = int;
    using DataBlock = std::map<
        MPI_CUSTOM_TAG, std::pair<
        std::variant<
            std::vector<char>, std::vector<int>,
            std::vector<uint32_t>, std::vector<uint64_t>,
            std::vector<float>, std::vector<double>
        >, bool> // bool is for static/dynamic array
    >;

    DataBlock buffer_data_;
    DataBlock element_data_;
    MPI_CUSTOM_TAG tag_ = CSTM_TAG_END;

public:
    PackedMessage() = default;

    PackedMessage(const MessageLayout layout) : 
        tag_(layout.tag()) 
    {
        for (const auto [key, value] : layout.buffer_layout_) {
            auto& [buffer, size] = value;
            std::visit([&](auto& buffer) {
                using Holder = std::decay_t<decltype(buffer)>;
                using T      = typename Holder::type;
                bool is_static = size != 0;
                buffer_data_.try_emplace(key, std::make_pair(std::vector<T>(size), is_static));
            }, buffer);
        }

        for (const auto [key, value] : layout.element_layout_) {
            auto& [buffer, size] = value;
            std::visit([&](auto& buffer) {
                using Holder = std::decay_t<decltype(buffer)>;
                using T      = typename Holder::type;
                bool is_static = size != 0;
                element_data_.try_emplace(key, std::make_pair(std::vector<T>(size), is_static));
            }, buffer);
        }
    }

    template<typename T> requires MessageSupportedTypes<T>
    std::vector<T>& get_buffer(const MPI_CUSTOM_TAG tag) {
        using VecT = std::vector<T>;
        
        auto it = buffer_data_.find(tag);
        massert(it != buffer_data_.end(), "Tag " + std::to_string(tag) + " not found");

        auto *vec = std::get_if<VecT>(&(it->second.first));
        massert(vec != nullptr, "Type mismatch for tag " + std::to_string(tag));
        return *vec;
    }

    template<typename T> requires MessageSupportedTypes<T>
    std::vector<T>& get_element(const MPI_CUSTOM_TAG tag) {
        using VecT = std::vector<T>;
        
        auto it = element_data_.find(tag);
        massert(it != element_data_.end(), "Tag " + std::to_string(tag) + " not found");

        auto *vec = std::get_if<VecT>(&(it->second.first));
        massert(vec != nullptr, "Type mismatch for tag " + std::to_string(tag));
        return *vec;
    }

    template<typename T> requires MessageSupportedTypes<T>
    const std::vector<T>& get_buffer(const MPI_CUSTOM_TAG tag) const {
        using VecT = std::vector<T>;
        
        auto it = buffer_data_.find(tag);
        massert(it != buffer_data_.end(), "Tag " + std::to_string(tag) + " not found");

        auto *vec = std::get_if<VecT>(&(it->second.first));
        massert(vec != nullptr, "Type mismatch for tag " + std::to_string(tag));
        return *vec;
    }

    template<typename T> requires MessageSupportedTypes<T>
    const std::vector<T>& get_element(const MPI_CUSTOM_TAG tag) const {
        using VecT = std::vector<T>;
        
        auto it = element_data_.find(tag);
        massert(it != element_data_.end(), "Tag " + std::to_string(tag) + " not found");

        auto *vec = std::get_if<VecT>(&(it->second.first));
        massert(vec != nullptr, "Type mismatch for tag " + std::to_string(tag));
        return *vec;
    }

    inline MPI_CUSTOM_TAG tag() const { return tag_; }

private:

    void pack_data(std::vector<char>& packed_data) const {
        std::size_t total_size = 0;

        for (const auto& [key, value] : element_data_) {
            auto& [buffer, is_static] = value;

            std::visit([&](auto& buf) {
                using VecT = std::decay_t<decltype(buf)>;
                using T    = typename VecT::value_type;
                MPI_Datatype mpi_type = get_mpi_type<T>();

                int size = 0;
                MPI_Pack_size(1, MPI_UINT32_T, MPI_COMM_WORLD, &size);
                total_size += size;

                MPI_Pack_size(static_cast<int>(buf.size()), mpi_type, 
                              MPI_COMM_WORLD, &size);
                total_size += size;
            }, buffer);
        }
        packed_data.resize(total_size);

        int position = 0;
        for (auto& [key, value] : element_data_) {
            auto& [buffer, is_static] = value;

            std::visit([&](auto& buf) {
                using VecT = std::decay_t<decltype(buf)>;
                using T    = typename VecT::value_type;
                MPI_Datatype mpi_type = get_mpi_type<T>();

                uint32_t size = static_cast<uint32_t>(buf.size());
                MPI_Pack(&size, 1, MPI_UINT32_T, packed_data.data(), 
                         total_size, &position, MPI_COMM_WORLD);
    
                if (size > 0) {
                    MPI_Pack(buf.data(), buf.size(), mpi_type, 
                         packed_data.data(), total_size, 
                         &position, MPI_COMM_WORLD);
                }
            }, buffer);
        }
    } 

    void unpack_data(std::vector<char>& packed_data) {
        int position = 0;
        uint32_t total_size = static_cast<uint32_t>(packed_data.size());

        for (auto& [key, value] : element_data_) {
            auto& [buffer, is_static] = value;

            uint32_t count = 0;
            MPI_Unpack(packed_data.data(), total_size, &position, 
                       &count, 1, MPI_UINT32_T, MPI_COMM_WORLD);

            std::visit([&](auto& buf) {
                using VecT = std::decay_t<decltype(buf)>;
                using T    = typename VecT::value_type;
                MPI_Datatype mpi_type = get_mpi_type<T>();

                buf.resize(count);
                MPI_Unpack(packed_data.data(), static_cast<int>(total_size), &position, 
                           buf.data(), static_cast<int>(count), mpi_type, MPI_COMM_WORLD);
            }, buffer);
        }
    }

    template<typename T>
    static constexpr MPI_Datatype get_mpi_type() noexcept {
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
    friend void sync_send(const int dest, const PackedMessage &message);
    friend int sync_recv(PackedMessage &message, int source);
};

}
