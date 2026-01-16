#!/bin/bash

set -e

BUILD_TYPES=("Debug" "Release" "RelWithDebInfo")
DATA_FOLDER="/home/bigmat18/Utils/VCLab_DS_PLY"
TARGET=100

TYPE=""
EXE=""

NUM_WORKER="3"
NUM_WORKER_THREADS="4"
NUM_MASTER_THREADS="4"

PROFILE_MASTER=false
PROFILE_WORKERS=false

usage() {
  echo "Usage:"
  echo "  $0 <type> <exe> -nw NUM -wt NUM -mt NUM [-pm] [-pw]"
  echo
  echo "  type : debug | release | reldeb"
  echo "  exe  : executable name (without path, e.g. myprog)"
  echo "  -nw  : number of MPI worker processes"
  echo "  -wt  : number of OpenMP threads per worker"
  echo "  -mt  : number of OpenMP threads for the master"
  echo "  -pm  : enable profiling on the master rank"
  echo "  -pw  : enable profiling on the worker ranks"
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
  case "$1" in
    debug)  echo "Debug" ;;
    release) echo "Release" ;;
    reldeb) echo "RelWithDebInfo" ;;
    *)
      echo "Unknown type $1" >&2
      exit 1
      ;;
  esac
}

TYPE_NAME="$(map_build_type "$TYPE")"

mkdir -p perf

MASTER_CMD=("build/${TYPE_NAME}/examples/${EXE}" "${DATA_FOLDER}" -n "${TARGET}")
WORKER_CMD=("build/${TYPE_NAME}/examples/${EXE}" "${DATA_FOLDER}" -n "${TARGET}")

if $PROFILE_MASTER; then
  MASTER_CMD=("perf" "record" "-g" "-o" "perf/${EXE}-master.perf.data" "${MASTER_CMD[@]}")
fi

if $PROFILE_WORKERS; then
  WORKER_CMD=("perf" "record" "-g" "-o" "perf/${EXE}-workers.perf.data" "${WORKER_CMD[@]}")
fi

time -p mpirun --bind-to none --report-bindings \
  -np 1 \
    -x OMP_NUM_THREADS="${NUM_MASTER_THREADS}" \
    "${MASTER_CMD[@]}" \
  : \
  -np "${NUM_WORKER}" \
    -x OMP_NUM_THREADS="${NUM_WORKER_THREADS}" \
    "${WORKER_CMD[@]}"
