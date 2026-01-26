#!/bin/bash
set -euo pipefail

INPUT_FOLDER=""
NUM_MESHES="1"
PARTITIONS="4"
TARGET_PERCENT="10"

TYPE=""
EXE=""

NUM_WORKER="4"
NUM_WORKER_THREADS="1"
NUM_MASTER_THREADS="4"


usage() {
  echo "Usage:"
  echo "  $0 <type> <exe> [options]"
  echo
  echo "Positional Arguments:"
  echo "  type : debug | release | reldeb"
  echo "  exe  : executable name (without path, e.g. myprog)"
  echo
  echo "Options:"
  echo "  -nw NUM  : number of MPI worker processes (Default: 4)"
  echo "  -wt NUM  : number of OpenMP threads per worker (Default: 1)"
  echo "  -mt NUM  : number of OpenMP threads for the master (Default: 4)"
  echo "  -i PATH  : input folder path"
  echo "  -p NUM   : number of partitions (Default: 4)"
  echo "  -n NUM   : number of meshes (Default: 500)"
  echo "  -t NUM   : target percentage (Default: 10)"
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
    debug)           echo "Debug" ;;
    release)         echo "Release" ;;
    reldeb|relwithdebinfo)
                     echo "RelWithDebInfo" ;;
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

if [ ! -x "$MASTER_BIN" ]; then
  echo "Error: Master binary not found or not executable: $MASTER_BIN" >&2
  exit 1
fi

if [ ! -x "$WORKER_BIN" ]; then
  echo "Error: Worker binary not found or not executable: $WORKER_BIN" >&2
  exit 1
fi

mkdir -p perf

MASTER_ARGS=(
  "${INPUT_FOLDER}"
  -t "${TARGET_PERCENT}"
  -n "${NUM_MESHES}"
  -p "${PARTITIONS}"
)
WORKER_ARGS=(
  -p "${PARTITIONS}"
)


echo "== Job info =="
echo "TYPE_NAME          = ${TYPE_NAME}"
echo "EXE                = ${EXE}"
echo "NUM_WORKER         = ${NUM_WORKER}"
echo "NUM_MASTER_THREADS = ${NUM_MASTER_THREADS}"
echo "NUM_WORKER_THREADS = ${NUM_WORKER_THREADS}"
echo


echo "[MASTER] Launching master with srun..."

srun \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --cpus-per-task="${NUM_MASTER_THREADS}" \
  --cpu-bind=cores \
  --export=ALL,OMP_NUM_THREADS="${NUM_MASTER_THREADS}" \
  "${MASTER_LAUNCH[@]}" &
MASTER_PID=$!


echo "[WORKER] Launching ${NUM_WORKER} workers (1 per nodo se possibile)..."

srun \
  --nodes="${NUM_WORKER}" \
  --ntasks="${NUM_WORKER}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${NUM_WORKER_THREADS}" \
  --cpu-bind=cores \
  --export=ALL,OMP_NUM_THREADS="${NUM_WORKER_THREADS}" \
  "${WORKER_LAUNCH[@]}"

wait "${MASTER_PID}"

echo "Job completed."
