#!/usr/bin/env bash
# Trainer+daemon side of the cross-node libfabric pull test.
# Runs on the trainer node. Starts the libfabric-enabled daemon (TCP control
# port), then runs the trainer_seal.py under the torch-base env to create+seal a
# rollout version, and leaves the daemon running until 'done' appears in WORKDIR.
set -uo pipefail

PACK=/apdcephfs_zwfy10/share_303541817/hunyuan/stellezhang/dev/bprl/offload_design_pack
NC=/apdcephfs_zwfy10/share_303541817/hunyuan/stellezhang/dev/bprl/north_comm
export LD_LIBRARY_PATH="$NC/build/3rdparty/libfabric/lib:/opt/gdrcopy/lib:/opt/conda/envs/bprl/lib:$PACK/build_lf/vendored_libs:${LD_LIBRARY_PATH:-}"

WORKDIR="$1"; TCP_PORT="$2"; SCHED="$3"; VERSION="$4"
# NIC the daemon's libfabric sender opens locally (its own device). Override per
# node; mlx5_8 is the GPU-affinity RDMA domain north_comm accepts on these hosts.
export OFLD_LIBFABRIC_NIC="${OFLD_LIBFABRIC_NIC:-mlx5_bond_1}"
SOCK="$WORKDIR/fo.sock"
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"

OFLD_LOG_LEVEL=2 "$PACK/build_lf/offloadd" --smoke-arena-mb 2048 --numa 0 --gpu 0 \
    --socket "$SOCK" --tcp-port "$TCP_PORT" >"$WORKDIR/daemon.log" 2>&1 &
DPID=$!
sleep 2
if ! kill -0 "$DPID" 2>/dev/null; then
    echo "DAEMON_FAILED"; sed -n '1,20p' "$WORKDIR/daemon.log"; exit 1
fi
echo "DAEMON_UP pid=$DPID sock=$SOCK port=$TCP_PORT"

cd "$PACK/python_api"
PYTHONPATH="$(pwd)" CUDA_VISIBLE_DEVICES=0 \
    /opt/conda/envs/torch-base/bin/python \
    "$PACK/tests/python/trainer_seal.py" "$SOCK" "$WORKDIR" "$SCHED" "$VERSION" \
    >"$WORKDIR/trainer.log" 2>&1
TRC=$?
echo "TRAINER_EXIT=$TRC"
sed -n '1,40p' "$WORKDIR/trainer.log"
kill "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null
rm -f "$SOCK"
echo "TRAINER_SIDE_DONE"
