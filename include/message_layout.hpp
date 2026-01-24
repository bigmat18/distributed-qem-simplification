#pragma once 
#include <map>
#include <cstdint>
#include <variant>
#include <utils.hpp>

namespace mpi {

#define CSTM_TAG_END 0

template <typename T>
concept MessageSupportedTypes =
    std::is_same_v<T, char>    ||
    std::is_same_v<T, int>     ||
    std::is_same_v<T, uint32_t>||
    std::is_same_v<T, uint64_t>||
    std::is_same_v<T, float>   ||
    std::is_same_v<T, double>;

struct MessageLayoutTypes {
    template<typename T> requires MessageSupportedTypes<T>
    struct Type { using type = T; };

    using Char   = Type<char>;
    using Int    = Type<int>;
    using U32    = Type<uint32_t>;
    using U64    = Type<uint64_t>;
    using Float  = Type<float>;
    using Double = Type<double>;
};

class MessageLayout {
    using MPI_CUSTOM_TAG = int;
    using SUPPORTED_DATA_TYPES = std::variant<
        MessageLayoutTypes::Char,
        MessageLayoutTypes::Int,
        MessageLayoutTypes::U32,
        MessageLayoutTypes::U64,
        MessageLayoutTypes::Float,
        MessageLayoutTypes::Double
    >;
    // Pair contain {SUPPORTED_DATA_TYPES, size}
    using LAYOUT_ELEMENT = std::pair<SUPPORTED_DATA_TYPES, uint32_t>;

    MPI_CUSTOM_TAG tag_;
    std::map<MPI_CUSTOM_TAG, LAYOUT_ELEMENT> element_layout_;
    std::map<MPI_CUSTOM_TAG, LAYOUT_ELEMENT> buffer_layout_;

public:
    MessageLayout(MPI_CUSTOM_TAG tag = CSTM_TAG_END) : tag_(tag) {};

    template<typename T, uint32_t size = 0> requires MessageSupportedTypes<T>
    MessageLayout& add_buffer(const MPI_CUSTOM_TAG tag) { 
        auto it = element_layout_.find(tag);
        massert(it == element_layout_.end(), "Tag present in elements");
        buffer_layout_.try_emplace(tag, std::make_pair(MessageLayoutTypes::Type<T>{}, size));
        return *this;
    }


    template<typename T, uint32_t size = 0> requires MessageSupportedTypes<T>
    MessageLayout& add_element(const MPI_CUSTOM_TAG tag) { 
        auto it = buffer_layout_.find(tag);
        massert(it == buffer_layout_.end(), "Tag present in elements");
        element_layout_.try_emplace(tag, std::make_pair(MessageLayoutTypes::Type<T>{}, size));
        return *this;
    }

    inline MPI_CUSTOM_TAG tag() const { return tag_; }

    friend class PackedMessage;
    friend class AsyncSend;

};

}
