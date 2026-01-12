#pragma once 
#include <map>
#include <cstdint>
#include <variant>

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

    MPI_CUSTOM_TAG tag_;
    std::map<MPI_CUSTOM_TAG, SUPPORTED_DATA_TYPES> layout_;

public:
    using value_type      = std::pair<MPI_CUSTOM_TAG, SUPPORTED_DATA_TYPES>;
    using iterator        = std::map<MPI_CUSTOM_TAG, SUPPORTED_DATA_TYPES>::iterator;
    using const_iterator  = std::map<MPI_CUSTOM_TAG, SUPPORTED_DATA_TYPES>::const_iterator;

    MessageLayout(MPI_CUSTOM_TAG tag = CSTM_TAG_END) : tag_(tag) {};

    template<typename T> requires MessageSupportedTypes<T>
    MessageLayout& add_buffer(const MPI_CUSTOM_TAG tag) { 
        layout_.try_emplace(tag, MessageLayoutTypes::Type<T>{});
        return *this;
    }

    inline MPI_CUSTOM_TAG tag() const { return tag_; }

    iterator begin() { return layout_.begin(); }
    iterator end()   { return layout_.end(); }

    const_iterator begin()  const { return layout_.begin(); }
    const_iterator end()    const { return layout_.end(); }
    const_iterator cbegin() const { return layout_.cbegin(); }
    const_iterator cend()   const { return layout_.cend(); }

    std::size_t size() const { return layout_.size(); }
};

}
