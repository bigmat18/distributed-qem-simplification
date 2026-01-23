#!/bin/bash

set -e

FLAGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 "
BUILD_TYPES=("Debug" "Release" "RelWithDebInfo")
ROOT_DIR="build"

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

setup() {
    local TYPE_NAME="$1"
    local CMAKE_TYPE=$(map_build_type "$TYPE_NAME")
    local DIR="$ROOT_DIR/$CMAKE_TYPE"

    echo "==> Configuring $CMAKE_TYPE in $DIR"
    mkdir -p "$DIR"
    cmake -B "$DIR" -DCMAKE_BUILD_TYPE="${CMAKE_TYPE}" . ${FLAGS}

    cd "$ROOT_DIR"
    ln -sf "$CMAKE_TYPE/compile_commands.json" compile_commands.json
}


if [ -z "$1" ]; then
    for T in debug release reldeb; do
        setup "$T"
    done
else
    setup "$1"
fi




