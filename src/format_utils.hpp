#pragma once 

#include <Eigen/Dense>
#include <format>

template <> 
struct std::formatter<Eigen::Vector3f> : std::formatter<std::string> {
    auto format(const Eigen::Vector3f p, auto& ctx) const {
        auto s = std::format("({}, {}, {})", p.x(), p.y(), p.z());
        return std::formatter<std::string>::format(s, ctx);
    }
};
