#!/usr/bin/env bash
# Rollout side of the cross-node libfabric pull test. Runs on the rollout node
# under the bprl env (Python 3.12, where north_comm imports). Reads job identity
# from WORKDIR/job.txt (written by the trainer), then pulls each sealed object
# over libfabric and verifies bytes against WORKDIR/obj_*.bin.
set -uo pipefail

PACK=/apdcephfs_zwfy10/share_303541817/hunyuan/stellezhang/dev/bprl/offload_design_pack
NC=/apdcephfs_zwfy10/share_303541817/hunyuan/stellezhang/dev/bprl/north_comm
export LD_LIBRARY_PATH="$NC/build/3rdparty/libfabric/lib:/opt/gdrcopy/lib:/opt/conda/envs/bprl/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$NC/python:${PYTHONPATH:-}"

WORKDIR="$1"; DAEMON_HOST="$2"; TCP_PORT="$3"; NIC="$4"

# Wait for the trainer to signal ready.
for _ in $(seq 1 120); do
    [ -f "$WORKDIR/ready" ] && break
    sleep 0.5
done
if [ ! -f "$WORKDIR/ready" ]; then echo "NO_READY_SIGNAL"; exit 1; fi

read JOB_ID LAUNCH_EPOCH VERSION < "$WORKDIR/job.txt"
echo "ROLLOUT job_id=$JOB_ID launch_epoch=$LAUNCH_EPOCH version=$VERSION host=$DAEMON_HOST port=$TCP_PORT nic=$NIC"

CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/bprl/bin/python \
    "$PACK/tests/python/rollout_libfabric_recv.py" \
    "$DAEMON_HOST" "$TCP_PORT" "$JOB_ID" "$LAUNCH_EPOCH" "$NIC" "$WORKDIR"
RC=$?
echo "ROLLOUT_EXIT=$RC"
touch "$WORKDIR/done"
exit $RC
