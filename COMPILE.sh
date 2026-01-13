#!/bin/bash

set -e

FLAGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
ROOT_DIR="build"
BUILD_TYPES=("Debug" "Release" "RelWithDebInfo")

map_build_type() {
    case "$1" in
        debug) echo "Debug" ;;
        release) echo "Release" ;;
        reldeb) echo "RelWithDebInfo" ;;
        *)
            echo "Unkown type $1" >&2
            exit 1
            ;;
    esac
}

build() {
    local TYPE_NAME="$1"
    local CMAKE_TYPE
    CMAKE_TYPE=$(map_build_type "$TYPE_NAME")
    local DIR="$ROOT_DIR/$CMAKE_TYPE"

    echo "==> Building $CMAKE_TYPE in $DIR"
    cmake --build "$DIR" --verbose -- -j
}

if [ -z "$1" ]; then
    for T in debug release reldeb; do
        build "$T"
    done
else
    build "$1"
fi
