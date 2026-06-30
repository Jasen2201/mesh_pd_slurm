#!/usr/bin/env bash
#SBATCH --job-name=minimax-m3-fp8-1node-1p1d-tp4
#SBATCH --account=amd-frameworks
#SBATCH --partition=amd-frameworks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=114
#SBATCH --gres=gpu:8
#SBATCH --nodelist=mia1-p02-g44
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/it-share/yajizhan/slurm_minimax_logs/minimax_m3_fp8_1node_1p1d_tp4-%j.out
#SBATCH --error=/it-share/yajizhan/slurm_minimax_logs/minimax_m3_fp8_1node_1p1d_tp4-%j.err
#
# Single-node 1P+1D PD-disaggregated benchmark for MiniMax-M3-MXFP4 FP8 on ATOM.
#   prefill: GPU 0-3 (TP=4, port 8010)
#   decode:  GPU 4-7 (TP=4, port 8020)
#   router:  port 8000
#
# Usage:
#   mkdir -p /it-share/yajizhan/slurm_minimax_logs
#   sbatch minimax_m3_fp8_1node_1p1d_tp4_slurm.sh

set -euo pipefail

# ======================== configuration ========================
MODEL_PATH="${MODEL_PATH:-/mnt/models/MiniMax-M3-MXFP4}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/atom-dev:MiniMax-M3-20260630}"
CONTAINER="${CONTAINER:-atom_mesh_minimax_m3_fp8_1node_1p1d_tp4_${SLURM_JOB_ID}}"

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
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"


ISL_LIST="${ISL_LIST:-8192}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-1,2,4,8,16,32,64,128,256}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"

WAIT_SERVER_TIMEOUT="${WAIT_SERVER_TIMEOUT:-1800}"
WAIT_ROUTER_TIMEOUT="${WAIT_ROUTER_TIMEOUT:-300}"

RUN_GSM8K="${RUN_GSM8K:-1}"
GSM8K_LIMIT="${GSM8K_LIMIT:-}"
GSM8K_NUM_FEWSHOT="${GSM8K_NUM_FEWSHOT:-5}"
GSM8K_NUM_CONCURRENT="${GSM8K_NUM_CONCURRENT:-64,256}"
GSM8K_BATCH_SIZE="${GSM8K_BATCH_SIZE:-65}"
GSM8K_MAX_GEN_TOKS="${GSM8K_MAX_GEN_TOKS:-16384}"

LOG_ROOT="${LOG_ROOT:-/it-share/yajizhan/slurm_minimax_logs/$(date +%m%d)_minimax_m3_fp8_1node_1p1d_tp4_${SLURM_JOB_ID}}"

# ======================== pre-flight ========================
echo "=== Job ${SLURM_JOB_ID} starting on $(hostname) at $(date -Is) ==="
NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

mkdir -p "${LOG_ROOT}"/{prefill,decode,router,bench,gsm8k,scripts}

# ======================== pre-cleanup ========================
echo "=== pre-cleanup: force-stopping all docker containers on ${NODE} ==="
srun --nodelist="$NODE" --nodes=1 --ntasks=1 --time=00:03:00 bash -c '
    hostname
    running=$(docker ps -q)
    if [[ -n "$running" ]]; then
        echo "  stopping $(echo "$running" | wc -l) running containers:"
        docker ps --format "    {{.ID}} {{.Names}}"
        docker stop -t 0 $running 2>&1 | sed "s/^/    /"
    else
        echo "  no running containers"
    fi
    sleep 2
    used=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | grep -v ": 0$" | head -5)
    if [[ -n "$used" ]]; then
        echo "  WARNING: some GPUs still have VRAM allocated:"
        echo "$used" | sed "s/^/    /"
    else
        echo "  all GPUs free"
    fi
' || echo "[pre-cleanup] WARNING: cleanup on $NODE had errors (non-fatal)"
echo "=== pre-cleanup done ==="

NODE_IP=$(srun --nodelist="$NODE" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")

cat <<INFO
=== Configuration (single-node 1P+1D FP8) ===
NODE    : ${NODE} (IP=${NODE_IP})
PREFILL : GPU 0-3 (TP=${PREFILL_TP}, port=${PREFILL_PORT})
DECODE  : GPU 4-7 (TP=${DECODE_TP}, port=${DECODE_PORT})
ROUTER  : ${NODE_IP}:${ROUTER_PORT}
MODEL   : ${MODEL_PATH}
IMAGE   : ${DOCKER_IMAGE}
BACKEND : atom (PD mooncake KV transfer, pure TP, single-node)
RUN_GSM8K  : ${RUN_GSM8K} (limit=${GSM8K_LIMIT:-all}, fewshot=${GSM8K_NUM_FEWSHOT})
ISL/OSL/CONC : ${ISL_LIST} / ${OSL} / ${CONC_LIST}
LOG_ROOT: ${LOG_ROOT}
=====================
INFO

# ======================== generate in-container scripts ========================
PREFILL_GPU_IDS="0,1,2,3"
DECODE_GPU_IDS="4,5,6,7"

cat > "${LOG_ROOT}/scripts/prefill.sh" <<'PREFILL_EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "[prefill] IP=${NODE_IP} TP=${PREFILL_TP} port=${PREFILL_PORT} GPU=${PREFILL_GPU_IDS}"

mkdir -p /workspace/logs

export HIP_VISIBLE_DEVICES=${PREFILL_GPU_IDS}
export PYTHONUNBUFFERED=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1
export ATOM_HOST_IP=${NODE_IP}
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 --server-port "${PREFILL_PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${PREFILL_TP}" \
    --kv_cache_dtype fp8 \
    --block-size "${BLOCK_SIZE}" \
    --gpu-memory-utilization "${MEM_FRACTION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","proxy_ip":"${NODE_IP}","handshake_port":${HANDSHAKE_PORT}}' \
    --no-enable_prefix_caching \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    ${EXTRA_SERVER_ARGS} \
    2>&1 | tee /workspace/logs/prefill.log
PREFILL_EOF

cat > "${LOG_ROOT}/scripts/decode.sh" <<'DECODE_EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "[decode] IP=${NODE_IP} TP=${DECODE_TP} port=${DECODE_PORT} GPU=${DECODE_GPU_IDS}"

mkdir -p /workspace/logs

export HIP_VISIBLE_DEVICES=${DECODE_GPU_IDS}
export PYTHONUNBUFFERED=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1
export ATOM_HOST_IP=${NODE_IP}
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 --server-port "${DECODE_PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${DECODE_TP}" \
    --kv_cache_dtype fp8 \
    --block-size "${BLOCK_SIZE}" \
    --gpu-memory-utilization "${MEM_FRACTION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","proxy_ip":"${NODE_IP}","handshake_port":${HANDSHAKE_PORT}}' \
    --cudagraph-capture-sizes "[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]" \
    --no-enable_prefix_caching \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    ${EXTRA_SERVER_ARGS} \
    2>&1 | tee /workspace/logs/decode.log
DECODE_EOF

cat > "${LOG_ROOT}/scripts/router.sh" <<'ROUTER_EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "[router] prefill=${NODE_IP}:${PREFILL_PORT} decode=${NODE_IP}:${DECODE_PORT} router=0.0.0.0:${ROUTER_PORT}"

mkdir -p /workspace/logs

/usr/local/bin/atomesh launch \
    --host 0.0.0.0 --port "${ROUTER_PORT}" \
    --pd-disaggregation \
    --prefill "http://${NODE_IP}:${PREFILL_PORT}" \
    --decode  "http://${NODE_IP}:${DECODE_PORT}" \
    --policy random \
    --backend atom \
    --log-dir /workspace/logs \
    --log-level info \
    --disable-health-check \
    --disable-circuit-breaker \
    --prometheus-port 29100 \
    2>&1 | tee /workspace/logs/router.log
ROUTER_EOF

cat > "${LOG_ROOT}/scripts/gsm8k.sh" <<'GSM8K_EOF'
#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="/workspace/gsm8k_results"

echo "[gsm8k] model=${MODEL_PATH} endpoint=http://127.0.0.1:${ROUTER_PORT}"
echo "[gsm8k] limit=${GSM8K_LIMIT:-all} fewshot=${GSM8K_NUM_FEWSHOT} concurrent=${GSM8K_NUM_CONCURRENT} batch_size=${GSM8K_BATCH_SIZE} max_gen_toks=${GSM8K_MAX_GEN_TOKS}"

if ! command -v lm_eval >/dev/null 2>&1; then
    echo "[gsm8k] installing lm-eval..."
    pip install 'lm-eval[api]'
fi

mkdir -p "${RESULT_DIR}"

LIMIT_ARG=""
if [[ -n "${GSM8K_LIMIT}" ]]; then
    LIMIT_ARG="--limit ${GSM8K_LIMIT}"
fi

IFS=',' read -ra GSM8K_CONCS <<< "${GSM8K_NUM_CONCURRENT}"
for GSM8K_CONC in "${GSM8K_CONCS[@]}"; do
    RUN_TAG="$(date +%Y%m%d%H%M%S)_gsm8k_minimax_m3_fp8_1node_1p1d_tp4_c${GSM8K_CONC}"
    echo ""
    echo "========================================="
    echo "[gsm8k] running with concurrent=${GSM8K_CONC}"
    echo "========================================="

    lm_eval --model local-chat-completions \
        --model_args "model=${MODEL_PATH},base_url=http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions,num_concurrent=${GSM8K_CONC},max_retries=3,max_gen_toks=${GSM8K_MAX_GEN_TOKS}" \
        --tasks gsm8k \
        --num_fewshot "${GSM8K_NUM_FEWSHOT}" \
        --batch_size "${GSM8K_BATCH_SIZE}" \
        --apply_chat_template \
        --fewshot_as_multiturn \
        ${LIMIT_ARG} \
        --output_path "${RESULT_DIR}/${RUN_TAG}"

    python3 -c "
from pathlib import Path
import json

result_dir = Path('${RESULT_DIR}/${RUN_TAG}')
json_files = list(result_dir.rglob('*.json')) if result_dir.is_dir() else []
if not json_files:
    print('[gsm8k] ERROR: no result JSON found')
    exit(1)

result_file = max(json_files, key=lambda p: p.stat().st_mtime)
data = json.load(open(result_file))
score = data.get('results', {}).get('gsm8k', {}).get('exact_match,flexible-extract', 'N/A')
print('=========================================')
print(f'[gsm8k] concurrent=${GSM8K_CONC} exact_match,flexible-extract = {score}')
print('=========================================')
print(json.dumps(data.get('results', {}), indent=2))
"
done

echo "[gsm8k] all runs done, results saved to ${RESULT_DIR}"
GSM8K_EOF

cat > "${LOG_ROOT}/scripts/benchmark.sh" <<'BENCH_EOF'
#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="/workspace/benchmark_results"

echo "[bench] model=${MODEL_PATH} endpoint=http://127.0.0.1:${ROUTER_PORT}"
echo "[bench] ISL=[${ISL_LIST}] OSL=${OSL} CONC=[${CONC_LIST}] ratio=${RANDOM_RANGE_RATIO}"

if [[ ! -d /tmp/sglang-benchmark/bench_serving ]]; then
    rm -rf /tmp/sglang-benchmark
    mkdir -p /tmp/sglang-benchmark
    git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/sglang-benchmark/bench_serving
fi

mkdir -p "${RESULT_DIR}"

IFS=',' read -ra ISLS <<< "${ISL_LIST}"
IFS=',' read -ra CONCS <<< "${CONC_LIST}"

for ISL in "${ISLS[@]}"; do
    for CONC in "${CONCS[@]}"; do
        RESULT_FILENAME="pd-atom-minimax-m3-fp8-1node-1p1d-tp4-${ISL}-${OSL}-${CONC}-${RANDOM_RANGE_RATIO}"
        echo ""
        echo "========================================="
        echo "[bench] ISL=${ISL} OSL=${OSL} CONC=${CONC}"
        echo "========================================="

        PYTHONDONTWRITEBYTECODE=1 python /tmp/sglang-benchmark/bench_serving/benchmark_serving.py \
            --model="${MODEL_PATH}" \
            --backend=vllm \
            --base-url="http://127.0.0.1:${ROUTER_PORT}" \
            --dataset-name=random \
            --random-input-len="${ISL}" \
            --random-output-len="${OSL}" \
            --random-range-ratio "${RANDOM_RANGE_RATIO}" \
            --num-prompts=$(( CONC * 10 )) \
            --max-concurrency="${CONC}" \
            --trust-remote-code \
            --num-warmups=$(( 2 * CONC )) \
            --request-rate=inf \
            --ignore-eos \
            --save-result \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --result-dir="${RESULT_DIR}" \
            --result-filename="${RESULT_FILENAME}.json"
    done
done

echo ""
echo "========================================="
echo "[bench] summary"
echo "========================================="

python3 -c "
from pathlib import Path
import json

result_dir = Path('${RESULT_DIR}')
json_files = sorted(result_dir.glob('pd-atom-minimax-m3-fp8-1node-1p1d-tp4-*.json'))
if not json_files:
    print('No result files found')
    exit(0)

print(f\"{'Config':<25} {'TTFT(ms)':>10} {'ITL(ms)':>10} {'Throughput(tok/s)':>18}\")
print('-' * 65)
for f in json_files:
    d = json.load(open(f))
    isl = d.get('random_input_len', '?')
    osl = d.get('random_output_len', '?')
    conc = d.get('max_concurrency', '?')
    ttft = d.get('mean_ttft_ms', 0)
    itl = d.get('mean_itl_ms', 0)
    tp = d.get('output_throughput', 0)
    print(f'{isl}/{osl} c={conc:<6} {ttft:>10.1f} {itl:>10.2f} {tp:>18.1f}')
"

echo "[bench] results saved to ${RESULT_DIR}"
BENCH_EOF

chmod +x "${LOG_ROOT}"/scripts/*.sh

for script in "${LOG_ROOT}"/scripts/*.sh; do
    sed -i \
        -e "s|\${NODE_IP}|${NODE_IP}|g" \
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

echo "[scripts] generated under ${LOG_ROOT}/scripts/"
ls -la "${LOG_ROOT}"/scripts/

# ======================== cleanup trap ========================
cleanup() {
    local rc=$?
    echo ""
    echo "=== cleanup (rc=${rc}) at $(date -Is) ==="
    docker logs "${CONTAINER}" > "${LOG_ROOT}/docker_$(hostname).log" 2>&1 || true
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    pkill -9 -f 'atom.entrypoints.openai_server' 2>/dev/null || true
    pkill -9 -f 'atomesh' 2>/dev/null || true
    echo "=== cleanup done; logs under ${LOG_ROOT} ==="
}
trap cleanup EXIT
trap 'echo "=== received signal, cleaning up ==="; exit 130' INT TERM

# ======================== helper ========================

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

# ======================== 1. start container ========================
echo "[container] starting on ${NODE}"
srun --nodelist="$NODE" --nodes=1 --ntasks=1 bash -lc "
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
        -v '${LOG_ROOT}/prefill':/workspace/logs \
        -v '${LOG_ROOT}/decode':/workspace/decode_logs \
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

# ======================== 2. start prefill server (detached) ========================
echo "[prefill] launching server on ${NODE} GPU 0-3"
srun --nodelist="$NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/prefill.sh'
"

# ======================== 3. start decode server (detached) ========================
echo "[decode] launching server on ${NODE} GPU 4-7"
srun --nodelist="$NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/decode.sh'
"

# ======================== 4. wait for servers ========================
wait_endpoint() {
    local url="$1" timeout="$2" name="$3"
    echo "[wait] ${name} -> ${url} (timeout ${timeout}s)"
    srun --nodelist="$NODE" --nodes=1 --ntasks=1 bash -lc "
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
    local base_url="$1" model="$2" timeout="$3" name="$4"
    echo "[wait-inference] ${name} -> ${base_url}/v1/completions (timeout ${timeout}s)"
    srun --nodelist="$NODE" --nodes=1 --ntasks=1 bash -lc "
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

wait_endpoint "http://${NODE_IP}:${PREFILL_PORT}/health" \
    "$WAIT_SERVER_TIMEOUT" "prefill-http"
wait_endpoint "http://${NODE_IP}:${DECODE_PORT}/health" \
    "$WAIT_SERVER_TIMEOUT" "decode-http"

# ======================== 5. start router (detached) ========================
echo "[router] launching on ${NODE}"
srun --nodelist="$NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/router.sh'
"

wait_endpoint "http://${NODE_IP}:${ROUTER_PORT}/v1/models" \
    "$WAIT_ROUTER_TIMEOUT" "router-http"

wait_inference_ready "http://${NODE_IP}:${ROUTER_PORT}" \
    "$MODEL_PATH" "$WAIT_SERVER_TIMEOUT" "router-pipeline"

# ======================== 6. run gsm8k accuracy (foreground, optional) ========================
if [[ "${RUN_GSM8K}" == "1" ]]; then
    echo ""
    echo "=== running GSM8K accuracy eval on ${NODE} ==="
    srun --nodelist="$NODE" --nodes=1 --ntasks=1 bash -lc "
        docker exec '${CONTAINER}' bash '${LOG_ROOT}/scripts/gsm8k.sh'
    "
else
    echo "=== skipping GSM8K (RUN_GSM8K=${RUN_GSM8K}) ==="
fi

# ======================== 7. run benchmark (foreground) ========================
echo ""
echo "=== running benchmark on ${NODE} ==="
srun --nodelist="$NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec '${CONTAINER}' bash '${LOG_ROOT}/scripts/benchmark.sh'
"

echo ""
echo "=== done at $(date -Is); results: ${LOG_ROOT}/bench  gsm8k: ${LOG_ROOT}/gsm8k ==="
