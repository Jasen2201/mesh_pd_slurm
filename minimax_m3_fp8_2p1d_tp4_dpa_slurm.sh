#!/usr/bin/env bash
#SBATCH --job-name=minimax-m3-fp8-2p1d-tp4-dpa
#SBATCH --account=amd-frameworks
#SBATCH --partition=amd-frameworks
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=114
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/it-share/yajizhan/slurm_minimax_logs/minimax_m3_fp8_2p1d_tp4_dpa-%j.out
#SBATCH --error=/it-share/yajizhan/slurm_minimax_logs/minimax_m3_fp8_2p1d_tp4_dpa-%j.err

#
# 2P+1D PD-disaggregated benchmark for MiniMax-M3-MXFP4 (FP8 online quant)
# on ATOM with DPA.
#   prefill: TP=4 (2 instances, 1 node each), decode: TP=4 (1 instance), --enable-dp-attention.
#   3 nodes total: node0=prefill-1, node1=prefill-2, node2=decode.
#   Each instance runs in its own container so ATOM's hardcoded port 29500 never conflicts.
#
# Usage:
#   mkdir -p /it-share/yajizhan/slurm_minimax_logs
#   sbatch minimax_m3_fp8_2p1d_tp4_dpa_slurm.sh

set -euo pipefail

# ── configuration ───────────────────────────────────────────────────────────────

MODEL_PATH="${MODEL_PATH:-/mnt/models/MiniMax-M3-MXFP4}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/atom-dev:MiniMax-M3-20260630}"
CONTAINER="${CONTAINER:-atom_mesh_minimax_m3_fp8_2p1d_tp4_dpa_${SLURM_JOB_ID}}"

PREFILL_TP="${PREFILL_TP:-4}"
DECODE_TP="${DECODE_TP:-4}"
PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-6301}"

MEM_FRACTION="${MEM_FRACTION:-0.8}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

ISL_LIST="${ISL_LIST:-8192}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-256,512,768,1024}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"

WAIT_SERVER_TIMEOUT="${WAIT_SERVER_TIMEOUT:-1800}"
WAIT_ROUTER_TIMEOUT="${WAIT_ROUTER_TIMEOUT:-300}"

RUN_GSM8K="${RUN_GSM8K:-1}"
GSM8K_LIMIT="${GSM8K_LIMIT:-}"
GSM8K_NUM_FEWSHOT="${GSM8K_NUM_FEWSHOT:-5}"
GSM8K_NUM_CONCURRENT="${GSM8K_NUM_CONCURRENT:-256,512,768,1024}"
GSM8K_BATCH_SIZE="${GSM8K_BATCH_SIZE:-65}"
GSM8K_MAX_GEN_TOKS="${GSM8K_MAX_GEN_TOKS:-16384}"

LOG_ROOT="${LOG_ROOT:-/it-share/yajizhan/slurm_minimax_logs/$(date +%m%d)_minimax_m3_fp8_2p1d_tp4_dpa_${SLURM_JOB_ID}}"

# ── hf overrides ───────────────────────────────────────────────────────────────


# ── pre-flight ──────────────────────────────────────────────────────────────────

IFS=',' read -ra NODES <<< "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd,)"
if [[ ${#NODES[@]} -ne 3 ]]; then
    echo "FATAL: expected 3 nodes, got ${#NODES[@]}: ${NODES[*]}"
    exit 1
fi

PREFILL_NODE_1="${NODES[0]}"
PREFILL_NODE_2="${NODES[1]}"
DECODE_NODE="${NODES[2]}"
ALL_NODES=("$PREFILL_NODE_1" "$PREFILL_NODE_2" "$DECODE_NODE")

mkdir -p "${LOG_ROOT}"/{prefill_1,prefill_2,decode,router,bench,gsm8k,scripts}

# ── pre-cleanup ─────────────────────────────────────────────────────────────────

for node in "${ALL_NODES[@]}"; do
    srun --nodelist="$node" --nodes=1 --ntasks=1 bash -lc "
        docker rm -f '${CONTAINER}' 2>/dev/null || true
    " &
done
wait

# ── IP detection ────────────────────────────────────────────────────────────────

PREFILL_IP_1=$(srun --nodelist="$PREFILL_NODE_1" --nodes=1 --ntasks=1 bash -lc "hostname -I | awk '{print \$1}'")
PREFILL_IP_2=$(srun --nodelist="$PREFILL_NODE_2" --nodes=1 --ntasks=1 bash -lc "hostname -I | awk '{print \$1}'")
DECODE_IP=$(srun --nodelist="$DECODE_NODE" --nodes=1 --ntasks=1 bash -lc "hostname -I | awk '{print \$1}'")

# ── configuration display ──────────────────────────────────────────────────────

cat <<INFO
=== Configuration ===
PREFILL-1 : ${PREFILL_NODE_1}  (IP=${PREFILL_IP_1}, TP=${PREFILL_TP}, port=${PREFILL_PORT})
PREFILL-2 : ${PREFILL_NODE_2}  (IP=${PREFILL_IP_2}, TP=${PREFILL_TP}, port=${PREFILL_PORT})
DECODE    : ${DECODE_NODE}     (IP=${DECODE_IP},    TP=${DECODE_TP},  port=${DECODE_PORT})
ROUTER    : ${PREFILL_IP_1}:${ROUTER_PORT}
MODEL     : ${MODEL_PATH}
IMAGE     : ${DOCKER_IMAGE}
BACKEND   : atom (PD mooncake KV transfer, FP8 online quant, DPA)
RUN_GSM8K : ${RUN_GSM8K} (limit=${GSM8K_LIMIT:-all}, fewshot=${GSM8K_NUM_FEWSHOT})
ISL/OSL/CONC : ${ISL_LIST} / ${OSL} / ${CONC_LIST}
LOG_ROOT  : ${LOG_ROOT}
=====================
INFO

# ── GPU IDs ─────────────────────────────────────────────────────────────────────

PREFILL_GPU_IDS=$(seq -s, 0 $((PREFILL_TP - 1)))
DECODE_GPU_IDS=$(seq -s, 0 $((DECODE_TP - 1)))

# ── prefill script (template) ──────────────────────────────────────────────────

cat > "${LOG_ROOT}/scripts/prefill.sh.tmpl" << 'PREFILL_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

export HIP_VISIBLE_DEVICES=${PREFILL_GPU_IDS}
export PYTHONUNBUFFERED=1
export HSA_NO_SCRATCH_RECLAIM=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1
export AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0
export ATOM_HOST_IP=__PREFILL_HANDSHAKE_IP__
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

python3 -m atom.entrypoints.openai_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 --server-port "${PREFILL_PORT}" \
    --trust-remote-code \
    -tp "${PREFILL_TP}" \
    --enable-dp-attention --enable-tbo prefill \
    --gpu-memory-utilization "${MEM_FRACTION}" \
    --kv_cache_dtype fp8 --block-size "${BLOCK_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*.gate.*", "*.block_sparse_moe.experts*"]}' \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","proxy_ip":"__PREFILL_HANDSHAKE_IP__","handshake_port":${HANDSHAKE_PORT}}' \
    --no-enable_prefix_caching \
    ${EXTRA_SERVER_ARGS} \
    2>&1 | tee /workspace/logs/prefill.log
PREFILL_SCRIPT

sed "s|__PREFILL_HANDSHAKE_IP__|${PREFILL_IP_1}|g" \
    "${LOG_ROOT}/scripts/prefill.sh.tmpl" > "${LOG_ROOT}/scripts/prefill_1.sh"
sed "s|__PREFILL_HANDSHAKE_IP__|${PREFILL_IP_2}|g" \
    "${LOG_ROOT}/scripts/prefill.sh.tmpl" > "${LOG_ROOT}/scripts/prefill_2.sh"
rm "${LOG_ROOT}/scripts/prefill.sh.tmpl"

# ── decode script ───────────────────────────────────────────────────────────────

cat > "${LOG_ROOT}/scripts/decode.sh" << 'DECODE_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

export HIP_VISIBLE_DEVICES=${DECODE_GPU_IDS}
export PYTHONUNBUFFERED=1
export HSA_NO_SCRATCH_RECLAIM=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1
export AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0
export ATOM_HOST_IP=${DECODE_IP}
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

python3 -m atom.entrypoints.openai_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 --server-port "${DECODE_PORT}" \
    --trust-remote-code \
    -tp "${DECODE_TP}" \
    --enable-dp-attention \
    --gpu-memory-utilization "${MEM_FRACTION}" \
    --kv_cache_dtype fp8 --block-size "${BLOCK_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${DECODE_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*.gate.*", "*.block_sparse_moe.experts*"]}' \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","proxy_ip":"${DECODE_IP}","handshake_port":${HANDSHAKE_PORT}}' \
    --cudagraph-capture-sizes "[1,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]" \
    --no-enable_prefix_caching \
    ${EXTRA_SERVER_ARGS} \
    2>&1 | tee /workspace/logs/decode.log
DECODE_SCRIPT

# ── router script ───────────────────────────────────────────────────────────────

cat > "${LOG_ROOT}/scripts/router.sh" << 'ROUTER_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

echo "[router] prefill-1=${PREFILL_IP_1}:${PREFILL_PORT}"
echo "[router] prefill-2=${PREFILL_IP_2}:${PREFILL_PORT}"
echo "[router] decode=${DECODE_IP}:${DECODE_PORT}"
echo "[router] router=0.0.0.0:${ROUTER_PORT}"

/usr/local/bin/atomesh launch \
    --host 0.0.0.0 --port "${ROUTER_PORT}" \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP_1}:${PREFILL_PORT}" \
    --prefill "http://${PREFILL_IP_2}:${PREFILL_PORT}" \
    --decode  "http://${DECODE_IP}:${DECODE_PORT}" \
    --policy random \
    --backend atom \
    --log-dir /workspace/logs \
    --log-level info \
    --disable-health-check \
    --disable-circuit-breaker \
    --prometheus-port 29100 \
    2>&1 | tee /workspace/logs/router.log
ROUTER_SCRIPT

# ── gsm8k script ───────────────────────────────────────────────────────────────

cat > "${LOG_ROOT}/scripts/gsm8k.sh" << 'GSM8K_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

IFS=',' read -ra CONC_ARRAY <<< "${GSM8K_NUM_CONCURRENT}"
for GSM8K_CONC in "${CONC_ARRAY[@]}"; do
    RUN_TAG="gsm8k_minimax_m3_fp8_2p1d_tp4_dpa_c${GSM8K_CONC}"
    echo ""
    echo "=== GSM8K run: ${RUN_TAG} ==="

    LIMIT_ARGS=""
    if [[ -n "${GSM8K_LIMIT}" ]]; then
        LIMIT_ARGS="--limit ${GSM8K_LIMIT}"
    fi

    lm_eval --model local-completions \
        --model_args "model=${MODEL_PATH},base_url=http://${PREFILL_IP_1}:${ROUTER_PORT}/v1/completions,num_concurrent=${GSM8K_CONC},max_retries=6,tokenized_requests=False" \
        --tasks gsm8k \
        --num_fewshot "${GSM8K_NUM_FEWSHOT}" \
        --batch_size "${GSM8K_BATCH_SIZE}" \
        --gen_kwargs "max_gen_toks=${GSM8K_MAX_GEN_TOKS}" \
        ${LIMIT_ARGS} \
        --output_path "/workspace/gsm8k_results/${RUN_TAG}" \
        --log_samples \
        2>&1 | tee "/workspace/gsm8k_results/${RUN_TAG}.log"
done
GSM8K_SCRIPT

# ── benchmark script ────────────────────────────────────────────────────────────

cat > "${LOG_ROOT}/scripts/benchmark.sh" << 'BENCH_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

IFS=',' read -ra ISL_ARRAY <<< "${ISL_LIST}"
IFS=',' read -ra CONC_ARRAY <<< "${CONC_LIST}"

for ISL in "${ISL_ARRAY[@]}"; do
    for CONC in "${CONC_ARRAY[@]}"; do
        RESULT_FILENAME="pd-atom-minimax-m3-fp8-2p1d-tp4-dpa-${ISL}-${OSL}-${CONC}-${RANDOM_RANGE_RATIO}"
        echo ""
        echo "=== bench ISL=${ISL} OSL=${OSL} CONC=${CONC} => ${RESULT_FILENAME} ==="

        python3 /workspace/benchmarks/benchmark_serving.py \
            --backend atom \
            --base-url "http://${PREFILL_IP_1}:${ROUTER_PORT}" \
            --endpoint /v1/completions \
            --model "${MODEL_PATH}" \
            --dataset-name random \
            --random-input-len "${ISL}" \
            --random-output-len "${OSL}" \
            --random-range-ratio "${RANDOM_RANGE_RATIO}" \
            --num-prompts "${CONC}" \
            --max-concurrency "${CONC}" \
            --result-dir /workspace/benchmark_results \
            --result-filename "${RESULT_FILENAME}" \
            2>&1 | tee "/workspace/benchmark_results/${RESULT_FILENAME}.log"
    done
done

echo ""
echo "=== summary ==="
python3 -c "
import json, glob, os
files = sorted(glob.glob('/workspace/benchmark_results/pd-atom-minimax-m3-fp8-2p1d-tp4-dpa-*.json'))
if not files:
    print('no result files found')
else:
    header = f'{\"ISL\":>6} {\"OSL\":>6} {\"CONC\":>6} {\"Tput(tok/s)\":>12} {\"TTFT_avg\":>10} {\"TTFT_p99\":>10} {\"TPOT_avg\":>10} {\"TPOT_p99\":>10} {\"ITL_avg\":>10} {\"ITL_p99\":>10}'
    print(header)
    print('-' * len(header))
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        isl = d.get('random_input_len', '?')
        osl = d.get('random_output_len', '?')
        conc = d.get('max_concurrency', d.get('num_prompts', '?'))
        tput = d.get('output_throughput', 0)
        ttft_avg = d.get('mean_ttft_ms', 0)
        ttft_p99 = d.get('p99_ttft_ms', 0)
        tpot_avg = d.get('mean_tpot_ms', 0)
        tpot_p99 = d.get('p99_tpot_ms', 0)
        itl_avg  = d.get('mean_itl_ms', 0)
        itl_p99  = d.get('p99_itl_ms', 0)
        print(f'{isl:>6} {osl:>6} {conc:>6} {tput:>12.1f} {ttft_avg:>10.1f} {ttft_p99:>10.1f} {tpot_avg:>10.1f} {tpot_p99:>10.1f} {itl_avg:>10.1f} {itl_p99:>10.1f}')
"
BENCH_SCRIPT

# ── chmod + variable substitution ──────────────────────────────────────────────

chmod +x "${LOG_ROOT}"/scripts/*.sh


for script in "${LOG_ROOT}"/scripts/*.sh; do
    sed -i \
        -e "s|\${PREFILL_IP_1}|${PREFILL_IP_1}|g" \
        -e "s|\${PREFILL_IP_2}|${PREFILL_IP_2}|g" \
        -e "s|\${DECODE_IP}|${DECODE_IP}|g" \
        -e "s|\${PREFILL_TP}|${PREFILL_TP}|g" \
        -e "s|\${DECODE_TP}|${DECODE_TP}|g" \
        -e "s|\${PREFILL_PORT}|${PREFILL_PORT}|g" \
        -e "s|\${DECODE_PORT}|${DECODE_PORT}|g" \
        -e "s|\${ROUTER_PORT}|${ROUTER_PORT}|g" \
        -e "s|\${HANDSHAKE_PORT}|${HANDSHAKE_PORT}|g" \
        -e "s|\${MODEL_PATH}|${MODEL_PATH}|g" \
        -e "s|\${MEM_FRACTION}|${MEM_FRACTION}|g" \
        -e "s|\${BLOCK_SIZE}|${BLOCK_SIZE}|g" \
        -e "s|\${MAX_MODEL_LEN}|${MAX_MODEL_LEN}|g" \
        -e "s|\${MAX_NUM_SEQS}|${MAX_NUM_SEQS}|g" \
        -e "s|\${DECODE_MAX_NUM_SEQS}|${DECODE_MAX_NUM_SEQS}|g" \
        -e "s|\${MAX_NUM_BATCHED_TOKENS}|${MAX_NUM_BATCHED_TOKENS}|g" \
        -e "s|\${PREFILL_GPU_IDS}|${PREFILL_GPU_IDS}|g" \
        -e "s|\${DECODE_GPU_IDS}|${DECODE_GPU_IDS}|g" \
        -e "s|\${EXTRA_SERVER_ARGS}|${EXTRA_SERVER_ARGS}|g" \
        -e "s|\${ISL_LIST}|${ISL_LIST}|g" \
        -e "s|\${OSL}|${OSL}|g" \
        -e "s|\${CONC_LIST}|${CONC_LIST}|g" \
        -e "s|\${RANDOM_RANGE_RATIO}|${RANDOM_RANGE_RATIO}|g" \
        -e "s|\${GSM8K_LIMIT}|${GSM8K_LIMIT}|g" \
        -e "s|\${GSM8K_NUM_FEWSHOT}|${GSM8K_NUM_FEWSHOT}|g" \
        -e "s|\${GSM8K_NUM_CONCURRENT}|${GSM8K_NUM_CONCURRENT}|g" \
        -e "s|\${GSM8K_BATCH_SIZE}|${GSM8K_BATCH_SIZE}|g" \
        -e "s|\${GSM8K_MAX_GEN_TOKS}|${GSM8K_MAX_GEN_TOKS}|g" \
        "$script"
done

# ── cleanup trap ────────────────────────────────────────────────────────────────

cleanup() {
    echo ""
    echo "=== cleanup: removing containers ==="
    for node in "${ALL_NODES[@]}"; do
        srun --nodelist="$node" --nodes=1 --ntasks=1 bash -lc "
            docker rm -f '${CONTAINER}' 2>/dev/null || true
        " &
    done
    wait
    echo "=== cleanup done ==="
}
trap cleanup EXIT

# ── helper functions ────────────────────────────────────────────────────────────

detect_nic_type() {
    if [[ -n "${MORI_NIC_TYPE:-}" ]]; then
        echo "$MORI_NIC_TYPE"
        return
    fi
    local bnxt=0 mlx5=0 ionic=0
    if [[ -d /sys/class/infiniband ]]; then
        for dev in /sys/class/infiniband/*; do
            local name
            name=$(basename "$dev")
            case "$name" in
                bnxt_re*) ((bnxt++)) ;;
                mlx5*)    ((mlx5++)) ;;
                ionic*)   ((ionic++)) ;;
                *)
                    local drv
                    drv=$(readlink -f "$dev/device/driver" 2>/dev/null || true)
                    drv=$(basename "$drv" 2>/dev/null || true)
                    case "$drv" in
                        bnxt*)  ((bnxt++)) ;;
                        mlx5*)  ((mlx5++)) ;;
                        ionic*) ((ionic++)) ;;
                    esac
                    ;;
            esac
        done
    fi
    if (( bnxt >= mlx5 && bnxt >= ionic && bnxt > 0 )); then
        echo "bnxt"
    elif (( ionic >= mlx5 && ionic > 0 )); then
        echo "ionic"
    else
        echo "mlx5"
    fi
}

find_host_ibverbs() {
    local candidates=(
        /usr/lib64/libibverbs.so.1
        /lib/x86_64-linux-gnu/libibverbs.so.1
        /usr/lib/x86_64-linux-gnu/libibverbs.so.1
    )
    for c in "${candidates[@]}"; do
        local resolved
        resolved=$(readlink -f "$c" 2>/dev/null || true)
        if [[ -f "$resolved" ]]; then
            echo "$resolved"
            return
        fi
    done
}

nic_mount_flags() {
    local nic_type="$1"
    local flags=()
    case "$nic_type" in
        bnxt)
            local host_ibverbs
            host_ibverbs=$(find_host_ibverbs)
            if [[ -n "$host_ibverbs" ]]; then
                flags+=(-v "$host_ibverbs:/lib/x86_64-linux-gnu/libibverbs.so.1")
            fi
            for lib in /usr/local/lib/libbnxt_re-rdmav*.so; do
                [[ -f "$lib" ]] && flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/libibverbs/$(basename "$lib")")
            done
            for lib in /usr/local/lib/libbnxt_re.so; do
                [[ -f "$lib" ]] && flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/$(basename "$lib")")
            done
            [[ -d /etc/libibverbs.d ]] && flags+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)
            ;;
        ionic)
            local host_ibverbs
            host_ibverbs=$(find_host_ibverbs)
            if [[ -n "$host_ibverbs" ]]; then
                flags+=(-v "$host_ibverbs:/lib/x86_64-linux-gnu/libibverbs.so.1")
            fi
            local ionic_dirs=(/usr/local/lib /usr/lib/x86_64-linux-gnu)
            for dir in "${ionic_dirs[@]}"; do
                for lib in "$dir"/libionic*.so; do
                    if [[ -f "$lib" ]]; then
                        local real; real=$(readlink -f "$lib")
                        [[ -f "$real" ]] && flags+=(-v "$real:$real")
                        flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/$(basename "$lib")")
                    fi
                done
            done
            local provider_dir=/usr/lib/x86_64-linux-gnu/libibverbs
            if [[ -d "$provider_dir" ]]; then
                for lib in "$provider_dir"/libionic-rdmav*.so; do
                    [[ -f "$lib" ]] && flags+=(-v "$lib:$lib")
                done
            fi
            [[ -d /etc/libibverbs.d ]] && flags+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)
            ;;
        mlx5) ;;
    esac
    echo "${flags[@]}"
}

launch_container() {
    local node="$1"
    local role="$2"
    echo "[${role}] starting container on ${node}"
    srun --nodelist="$node" --nodes=1 --ntasks=1 bash -lc "
        set -euo pipefail

        $(declare -f detect_nic_type find_host_ibverbs nic_mount_flags)
        NIC_TYPE=\$(detect_nic_type)
        echo \"[docker] NIC type detected: \${NIC_TYPE} on \$(hostname)\"
        read -ra NIC_MOUNTS <<< \"\$(nic_mount_flags \"\${NIC_TYPE}\")\"
        if [[ \${#NIC_MOUNTS[@]} -gt 0 ]]; then
            echo \"[docker] RDMA mounts: \${NIC_MOUNTS[*]}\"
        else
            echo \"[docker] no out-of-tree RDMA mounts needed\"
        fi

        docker rm -f '${CONTAINER}' 2>/dev/null || true
        docker pull '${DOCKER_IMAGE}'
        docker run -d --name '${CONTAINER}' \
            --network host --ipc host --privileged \
            --device /dev/kfd --device /dev/dri \
            --device /dev/infiniband \
            --group-add video \
            --cap-add IPC_LOCK --cap-add NET_ADMIN \
            --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:524288 \
            --shm-size 128G \
            -v /mnt:/mnt \
            -v /data:/data \
            -v /it-share:/it-share \
            -v '${LOG_ROOT}/${role}':/workspace/logs \
            -v '${LOG_ROOT}/bench':/workspace/benchmark_results \
            -v '${LOG_ROOT}/gsm8k':/workspace/gsm8k_results \
            \"\${NIC_MOUNTS[@]}\" \
            '${DOCKER_IMAGE}' sleep infinity
        docker inspect -f '{{.State.Status}}' '${CONTAINER}'

        docker exec '${CONTAINER}' bash -c '
            sysctl -w net.core.somaxconn=4096 2>/dev/null || true
            sysctl -w net.ipv4.tcp_max_syn_backlog=4096 2>/dev/null || true
        '
        echo \"[docker] tuned TCP backlog on \$(hostname)\"
    "
}

wait_endpoint() {
    local node="$1" url="$2" timeout="$3" name="$4"
    echo "[wait] ${name} -> ${url} (timeout ${timeout}s)"
    srun --nodelist="$node" --nodes=1 --ntasks=1 bash -lc "
        deadline=\$(( \$(date +%s) + ${timeout} ))
        while ! curl -sf '${url}' >/dev/null 2>&1; do
            if [[ \$(date +%s) -ge \$deadline ]]; then
                echo '[wait][FAIL] ${name} not ready after ${timeout}s'
                exit 1
            fi
            sleep 10
        done
        echo '[wait][OK] ${name} ready'
    "
}

wait_inference_ready() {
    local node="$1" base_url="$2" model="$3" timeout="$4" name="$5"
    echo "[wait-inference] ${name} -> ${base_url}/v1/completions (timeout ${timeout}s)"
    srun --nodelist="$node" --nodes=1 --ntasks=1 bash -lc "
        deadline=\$(( \$(date +%s) + ${timeout} ))
        attempt=0
        while true; do
            attempt=\$((attempt + 1))
            resp=\$(curl -sS -m 120 -X POST '${base_url}/v1/completions' \
                -H 'Content-Type: application/json' \
                -d '{\"model\":\"${model}\",\"prompt\":\"hi\",\"max_tokens\":4,\"temperature\":0}' 2>&1 || true)
            text_len=\$(echo \"\$resp\" | python3 -c 'import sys,json
try:
    d=json.loads(sys.stdin.read())
    print(len(d.get(\"choices\",[{}])[0].get(\"text\",\"\")))
except Exception:
    print(0)' 2>/dev/null || echo 0)
            if [[ \"\$text_len\" -gt 0 ]]; then
                echo \"[wait-inference][OK] ${name} ready (attempt #\${attempt}, text_len=\${text_len})\"
                exit 0
            fi
            if [[ \$(date +%s) -ge \$deadline ]]; then
                echo \"[wait-inference][FAIL] ${name} not ready after ${timeout}s (attempts=\${attempt})\"
                echo \"[wait-inference] last response (truncated): \${resp:0:500}\"
                exit 1
            fi
            sleep 15
        done
    "
}

# ── step 1: launch containers ──────────────────────────────────────────────────

launch_container "$PREFILL_NODE_1"  prefill_1
launch_container "$PREFILL_NODE_2"  prefill_2
launch_container "$DECODE_NODE"     decode

# ── step 2: launch prefill servers ──────────────────────────────────────────────

echo "[prefill-1] launching server on ${PREFILL_NODE_1}"
srun --nodelist="$PREFILL_NODE_1" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/prefill_1.sh'
"
echo "[prefill-2] launching server on ${PREFILL_NODE_2}"
srun --nodelist="$PREFILL_NODE_2" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/prefill_2.sh'
"

# ── step 3: launch decode server ───────────────────────────────────────────────

echo "[decode] launching server on ${DECODE_NODE}"
srun --nodelist="$DECODE_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/decode.sh'
"

# ── step 4: wait for all servers ───────────────────────────────────────────────

wait_endpoint "$PREFILL_NODE_1" "http://${PREFILL_IP_1}:${PREFILL_PORT}/health" \
    "$WAIT_SERVER_TIMEOUT" "prefill-1-http"
wait_endpoint "$PREFILL_NODE_2" "http://${PREFILL_IP_2}:${PREFILL_PORT}/health" \
    "$WAIT_SERVER_TIMEOUT" "prefill-2-http"
wait_endpoint "$DECODE_NODE"    "http://${DECODE_IP}:${DECODE_PORT}/health" \
    "$WAIT_SERVER_TIMEOUT" "decode-http"

# ── step 5: launch router ──────────────────────────────────────────────────────

echo "[router] launching on ${PREFILL_NODE_1}"
srun --nodelist="$PREFILL_NODE_1" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/router.sh'
"

wait_endpoint "$PREFILL_NODE_1" "http://${PREFILL_IP_1}:${ROUTER_PORT}/v1/models" \
    "$WAIT_ROUTER_TIMEOUT" "router-http"

wait_inference_ready "$PREFILL_NODE_1" "http://${PREFILL_IP_1}:${ROUTER_PORT}" \
    "$MODEL_PATH" "$WAIT_SERVER_TIMEOUT" "router-pipeline"

# ── step 6: GSM8K (optional) ───────────────────────────────────────────────────

if [[ "${RUN_GSM8K}" == "1" ]]; then
    echo ""
    echo "=== running GSM8K accuracy eval on ${PREFILL_NODE_1} ==="
    srun --nodelist="$PREFILL_NODE_1" --nodes=1 --ntasks=1 bash -lc "
        docker exec '${CONTAINER}' bash '${LOG_ROOT}/scripts/gsm8k.sh'
    "
else
    echo "=== skipping GSM8K (RUN_GSM8K=${RUN_GSM8K}) ==="
fi

# ── step 7: benchmark ──────────────────────────────────────────────────────────

echo ""
echo "=== running benchmark on ${PREFILL_NODE_1} ==="
srun --nodelist="$PREFILL_NODE_1" --nodes=1 --ntasks=1 bash -lc "
    docker exec '${CONTAINER}' bash '${LOG_ROOT}/scripts/benchmark.sh'
"

echo ""
echo "=== done at $(date -Is); results: ${LOG_ROOT}/bench  gsm8k: ${LOG_ROOT}/gsm8k ==="
