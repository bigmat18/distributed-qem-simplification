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

update_compile_commands_link() {
    local CMAKE_TYPE="$1"
    mkdir -p "$ROOT_DIR"
    (
        cd "$ROOT_DIR"
        ln -sf "$CMAKE_TYPE/compile_commands.json" compile_commands.json
    )
}

setup_one() {
    local TYPE_NAME="$1"
    local CMAKE_TYPE
    CMAKE_TYPE=$(map_build_type "$TYPE_NAME")
    local DIR="$ROOT_DIR/$CMAKE_TYPE"

    echo "==> Configuring $CMAKE_TYPE in $DIR"
    mkdir -p "$DIR"
    cmake -B "$DIR" -DCMAKE_BUILD_TYPE="$CMAKE_TYPE" . ${FLAGS}

    update_compile_commands_link "$CMAKE_TYPE"
}

build_one() {
    local TYPE_NAME="$1"
    local CMAKE_TYPE
    CMAKE_TYPE=$(map_build_type "$TYPE_NAME")
    local DIR="$ROOT_DIR/$CMAKE_TYPE"

    echo "==> Building $CMAKE_TYPE in $DIR"
    cmake --build "$DIR" --verbose -- -j
}

setup_all() {
    for T in debug release reldeb; do
        setup_one "$T"
    done
}

build_all() {
    for T in debug release reldeb; do
        build_one "$T"
    done
}

clean() {
    rm -rf build/ out/
}

help() {
    cat <<EOF
Usage:
  $0 setup {debug|release|reldeb|all}
  $0 build {debug|release|reldeb|all}
  $0 clean
EOF
}

CMD="$1"
TYPE="$2"

case "$CMD" in
    setup)
        case "$TYPE" in
            debug|release|reldeb) setup_one "$TYPE" ;;
            all) setup_all ;;
            *) help ; exit 1 ;;
        esac
        ;;
    build)
        case "$TYPE" in
            debug|release|reldeb) build_one "$TYPE" ;;
            all) build_all ;;
            *) help ; exit 1 ;;
        esac
        ;;
    clean)
        clean
        ;;
    *)
        help
        exit 1
        ;;
esac
