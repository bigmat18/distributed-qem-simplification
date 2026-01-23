#!/bin/bash

set -euo pipefail

BUILD_TYPES=("Debug" "Release" "RelWithDebInfo")
INPUT_FOLDER=""
NUM_MESHES="500"
PARTITIONS="4"
TARGET_PERCENT="10"

TYPE=""
EXE=""

NUM_WORKER="4"
NUM_WORKER_THREADS="1"
NUM_MASTER_THREADS="4"

PROFILE_MASTER=false
PROFILE_WORKERS=false

usage() {
  echo "Usage:"
  echo "  $0 <type> <exe> [options]"
  echo
  echo "Positional Arguments:"
  echo "  type : debug | release | reldeb"
  echo "  exe  : executable name (without path, e.g. myprog)"
  echo
  echo "Options:"
  echo "  -nw NUM  : number of MPI worker processes (Default: 3)"
  echo "  -wt NUM  : number of OpenMP threads per worker (Default: 1)"
  echo "  -mt NUM  : number of OpenMP threads for the master (Default: 1)"
  echo "  -i PATH  : input folder path"
  echo "  -p NUM   : number of partitions (Default: 4)"
  echo "  -n NUM   : number of meshes (Default: 500)"
  echo "  -t NUM   : target percentage (Default: 10)"
  echo "  -pm      : enable profiling on the master rank"
  echo "  -pw      : enable profiling on the worker ranks"
  echo "  -h       : show this help message"
  exit 1
}

if [ "$#" -lt 2 ]; then
  echo "Error: <type> and <exe> are required" >&2
  usage
fi

TYPE="$1"
EXE="$2"
shift 2

while [ "$#" -gt 0 ]; do
  case "$1" in
    -nw)
      NUM_WORKER="$2"
      shift 2
      ;;
    -wt)
      NUM_WORKER_THREADS="$2"
      shift 2
      ;;
    -mt)
      NUM_MASTER_THREADS="$2"
      shift 2
      ;;
    -i)
      INPUT_FOLDER="$2"
      shift 2
      ;;
    -p)
      PARTITIONS="$2"
      shift 2
      ;;
    -n)
      NUM_MESHES="$2"
      shift 2
      ;;
    -t)
      TARGET_PERCENT="$2"
      shift 2
      ;;
    -pm)
      PROFILE_MASTER=true
      shift 1
      ;;
    -pw)
      PROFILE_WORKERS=true
      shift 1
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

map_build_type() {
  case "${1,,}" in
    debug)  echo "Debug" ;;
    release) echo "Release" ;;
    reldeb|relwithdebinfo) echo "RelWithDebInfo" ;;
    *)
      echo "Error: Unknown build type '$1'" >&2
      echo "Available types: debug, release, reldeb" >&2
      exit 1
      ;;
  esac
}

TYPE_NAME="$(map_build_type "$TYPE")"

BASE_PATH="build/${TYPE_NAME}/examples/${EXE}"
MASTER_BIN="${BASE_PATH}/master"
WORKER_BIN="${BASE_PATH}/worker"

if [ ! -f "$MASTER_BIN" ] || [ ! -f "$WORKER_BIN" ]; then
    echo "Error: Executables not found." >&2
    echo "  Expected Master: $MASTER_BIN" >&2
    echo "  Expected Worker: $WORKER_BIN" >&2
    exit 1
fi

mkdir -p perf

MASTER_ARGS=("${INPUT_FOLDER}" -t "${TARGET_PERCENT}" -n "${NUM_MESHES}" -p "${PARTITIONS}")
WORKER_ARGS=(-p "${PARTITIONS}")

if $PROFILE_MASTER; then
  MASTER_CMD=("perf" "record" "-g" "-o" "perf/${EXE}-master.perf.data" "${MASTER_BIN}" "${MASTER_ARGS[@]}")
else
  MASTER_CMD=("${MASTER_BIN}" "${MASTER_ARGS[@]}")
fi

if $PROFILE_WORKERS; then
  WORKER_CMD=("perf" "record" "-g" "-o" "perf/${EXE}-worker.perf.data" "${WORKER_BIN}" "${WORKER_ARGS[@]}")
else
  WORKER_CMD=("${WORKER_BIN}" "${WORKER_ARGS[@]}")
fi


time -p mpirun --use-hwthread-cpus --bynode --bind-to none --report-bindings \
  -np 1 \
    -x OMP_NUM_THREADS="${NUM_MASTER_THREADS}" \
    "${MASTER_CMD[@]}" \
  : \
  -np "${NUM_WORKER}" \
    -x OMP_NUM_THREADS="${NUM_WORKER_THREADS}" \
    "${WORKER_CMD[@]}"
