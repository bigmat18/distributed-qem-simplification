#!/bin/bash

FLAGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

all() {
    set -e

    BUILD_TYPES=("Debug" "Release" "RelWithDebInfo")
    ROOT_DIR="build"
    
    mkdir -p "$ROOT_DIR"
    
    for TYPE in "${BUILD_TYPES[@]}"; do
        DIR="$ROOT_DIR/$TYPE"
        echo "==> Configuring $TYPE in $DIR"
        cmake -B "$DIR" -DCMAKE_BUILD_TYPE=$TYPE . ${FLAGS}
    done
    
    cd build
    ln -s Debug/compile_commands.json compile_commands.json
    cd ..
}

debug() {
    cmake --build build/Debug --verbose -- -j 
}

release() {
    cmake --build build/Release --verbose -- -j 
}

reldeb() {
    cmake --build build/RelWithDebInfo --verbose -- -j 
}

clean() {
    rm -rf build/ out/
}

help() {
    echo "Usage: $0 {all|debug|release|reldeb|connect|clean}"
}

case "$1" in
    all) all ;;
    debug) debug ;;
    release) release ;;
    reldeb) reldeb ;;
    connect) connect ;;
    clean) clean ;;
    *) help ;;
esac
