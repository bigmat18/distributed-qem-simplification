#pragma once 

#include <cstdint>
#include <variant>
#include <vector>
#include <map>

#include <utils.hpp>
#include <mpi.h>

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

    DataBlock data_;
    MPI_CUSTOM_TAG tag_ = CSTM_TAG_END;

public:
    PackedMessage() = default;

    PackedMessage(const MessageLayout layout) : 
        tag_(layout.tag()) 
    {
        for (const auto [key, value] : layout) {
            auto& [buffer, size] = value;
            std::visit([&](auto& buffer) {
                using Holder = std::decay_t<decltype(buffer)>;
                using T      = typename Holder::type;
                bool is_static = size != 0;
                data_.try_emplace(key, std::make_pair(std::vector<T>(size), is_static));
            }, buffer);
        }
    }

    template<typename T> requires MessageSupportedTypes<T>
    std::vector<T>& get_buffer(const MPI_CUSTOM_TAG tag) {
        using VecT = std::vector<T>;
        
        auto it = data_.find(tag);
        massert(it != data_.end(), "Tag " + std::to_string(tag) + " not found");

        auto *vec = std::get_if<VecT>(&(it->second.first));
        massert(vec != nullptr, "Type mismatch for tag " + std::to_string(tag));
        return *vec;
    }

    template<typename T> requires MessageSupportedTypes<T>
    const std::vector<T>& get_buffer(const MPI_CUSTOM_TAG tag) const {
        using VecT = std::vector<T>;
        
        auto it = data_.find(tag);
        massert(it != data_.end(), "Tag " + std::to_string(tag) + " not found");

        auto *vec = std::get_if<VecT>(&(it->second.first));
        massert(vec != nullptr, "Type mismatch for tag " + std::to_string(tag));
        return *vec;
    }

    bool is_static(const MPI_CUSTOM_TAG tag) {
        auto it = data_.find(tag);
        massert(it != data_.end(), "Tag " + std::to_string(tag) + " not found");
        return it->second.second;
    }

    inline MPI_CUSTOM_TAG tag() const { return tag_; }

    auto begin() noexcept { return data_.begin(); }
    auto end()   noexcept { return data_.end(); }

    auto begin()  const noexcept { return data_.begin(); }
    auto end()    const noexcept { return data_.end(); }
    auto cbegin() const noexcept { return data_.cbegin(); }
    auto cend()   const noexcept { return data_.cend();}

    std::size_t size() const noexcept { return data_.size(); }

private:

    void pack_data(std::vector<char>& packed_data) {
        std::size_t total_size = 0;
        int pack_overhead = 0;

        for (const auto& [key, value] : data_) {
            auto& [buffer, is_static] = value;

            std::visit([&](auto& buf) {
                using VecT = std::decay_t<decltype(buf)>;
                using T    = typename VecT::value_type;
                MPI_Datatype mpi_type = get_mpi_type<T>();

                if (!is_static && !buf.empty()) {
                    MPI_Pack_size(1, MPI_UINT32_T, MPI_COMM_WORLD, &pack_overhead);
                    total_size += pack_overhead;
                }

                if (!buf.empty()) {
                    MPI_Pack_size(static_cast<int>(buf.size()), get_mpi_type<T>(), 
                                  MPI_COMM_WORLD, &pack_overhead);
                    total_size += pack_overhead;
                }

            }, buffer);
        }
        packed_data.resize(total_size);

        int position = 0;
        for (auto& [key, value] : data_) {
            auto& [buffer, is_static] = value;

            std::visit([&](auto& buf) {
                using VecT = std::decay_t<decltype(buf)>;
                using T    = typename VecT::value_type;
                MPI_Datatype mpi_type = get_mpi_type<T>();

                if (!is_static && !buf.empty()) {
                    uint32_t s = static_cast<uint32_t>(buf.size());
                    MPI_Pack(&s, 1, MPI_UINT32_T, packed_data.data(), 
                             total_size, &position, MPI_COMM_WORLD);
                }
    
                if (!buf.empty()) {
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

        for (auto& [key, value] : data_) {
            auto& [buffer, is_static] = value;

            uint32_t size = 0;
            MPI_Unpack(packed_data.data(), total_size, &position, 
                       &size, 1, MPI_UINT32_T, MPI_COMM_WORLD); 

            std::visit([&](auto& buf) {
                using VecT = std::decay_t<decltype(buf)>;
                using T    = typename VecT::value_type;
                MPI_Datatype mpi_type = get_mpi_type<T>();

                uint32_t count = 0;

                if (is_static) {
                    count = static_cast<uint32_t>(buf.size()); 
                } else {
                    MPI_Unpack(packed_data.data(), total_size, &position, 
                               &count, 1, MPI_UINT32_T, MPI_COMM_WORLD);

                    if (buf.size() != count)
                        buf.resize(count);
                }

                if (count > 0) {
                     MPI_Unpack(packed_data.data(), total_size, &position, 
                                buf.data(), count, mpi_type, MPI_COMM_WORLD);
                }

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
