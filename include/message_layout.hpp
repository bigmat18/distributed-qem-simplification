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
    // Pair contain {SUPPORTED_DATA_TYPES, size}
    using LAYOUT_ELEMENT = std::pair<SUPPORTED_DATA_TYPES, uint32_t>;

    MPI_CUSTOM_TAG tag_;
    std::map<MPI_CUSTOM_TAG, LAYOUT_ELEMENT> layout_;

public:
    using value_type      = std::pair<MPI_CUSTOM_TAG, LAYOUT_ELEMENT>;
    using iterator        = std::map<MPI_CUSTOM_TAG, LAYOUT_ELEMENT>::iterator;
    using const_iterator  = std::map<MPI_CUSTOM_TAG, LAYOUT_ELEMENT>::const_iterator;

    MessageLayout(MPI_CUSTOM_TAG tag = CSTM_TAG_END) : tag_(tag) {};

    template<typename T, uint32_t size = 0> requires MessageSupportedTypes<T>
    MessageLayout& add_buffer(const MPI_CUSTOM_TAG tag) { 
        layout_.try_emplace(tag, std::make_pair(MessageLayoutTypes::Type<T>{}, size));
        return *this;
    }

    inline MPI_CUSTOM_TAG tag() const { return tag_; }

    iterator begin() noexcept { return layout_.begin(); }
    iterator end()   noexcept { return layout_.end(); }

    const_iterator begin()  const noexcept { return layout_.begin(); }
    const_iterator end()    const noexcept { return layout_.end(); }
    const_iterator cbegin() const noexcept { return layout_.cbegin(); }
    const_iterator cend()   const noexcept { return layout_.cend(); }

    std::size_t size() const noexcept { return layout_.size(); }
};

}
