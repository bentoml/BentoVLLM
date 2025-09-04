#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# LMCache Disaggregated Prefill - XpYd (NIXL/UCX) on vLLM v1
# =============================================================================
# Auto-alloc ports from BASE_* envs; expand GPU ranges [a,b) / [a,b];
# if only PREFILL_GPUS is set, DECODE_GPUS := (all GPUs) \ PREFILL_GPUS.
# =============================================================================

# ---------- Tunables ----------
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}

# Proxy (HTTP) and NIXL (intra-node) plumbing
PROXY_PORT=${PROXY_PORT:-9100}
NIXL_PROXY_PORT=${NIXL_PROXY_PORT:-7500}
NIXL_PEER_INIT_BASE=${NIXL_PEER_INIT_BASE:-7300}
NIXL_PEER_ALLOC_BASE=${NIXL_PEER_ALLOC_BASE:-7400}

# Auto-port bases (sequential allocation)
BASE_PREFILL_PORT=${BASE_PREFILL_PORT:-7100}
BASE_DECODE_PORT=${BASE_DECODE_PORT:-7200}

# Orchestration
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export PYTHONHASHSEED

# UCX and vLLM knobs
export UCX_TLS=${UCX_TLS:-cuda_ipc,cuda_copy,tcp}
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1

# ---------- Detect whether user actually set these envs ----------
USER_SET_PREFILL=$([[ -v PREFILL_GPUS ]] && echo 1 || echo 0)
USER_SET_DECODE=$([[ -v DECODE_GPUS ]] && echo 1 || echo 0)
USER_SET_PPORTS=$([[ -v PREFILL_PORTS ]] && echo 1 || echo 0)
USER_SET_DPORTS=$([[ -v DECODE_PORTS ]] && echo 1 || echo 0)

# Defaults only if not set by user (used when we don't compute complement)
DEFAULT_PREFILL_GPUS="0,1"
DEFAULT_DECODE_GPUS="2,3"

# ---------- Housekeeping ----------
cd "$(dirname "${BASH_SOURCE[0]}")"
PIDS=()

require() {
	command -v "$1" >/dev/null 2>&1 || {
		echo "Missing required binary: $1"
		exit 1
	}
}
pyreq() {
	python3 - "$1" <<'PY' >/dev/null 2>&1 || exit 1
import importlib, sys
mod = sys.argv[1]
importlib.import_module(mod)
PY
	rc=$?
	if [[ $rc -ne 0 ]]; then
		echo "Missing Python module: $1"
		exit 1
	fi
}

check_required_files() {
	local files=("disagg_proxy_server.py")
	for f in "${files[@]}"; do
		[[ -f "$f" ]] || {
			echo "Required file $f not found in $(pwd)"
			exit 1
		}
	done
}
check_hf_token() {
	if [[ -z "${HF_TOKEN:-}" || "$HF_TOKEN" != hf_* ]]; then
		echo "HF_TOKEN not set/invalid; export a valid Hugging Face token."
		exit 1
	fi
}

# ---------- Small utils ----------
join_by() {
	local IFS="$1"
	shift
	echo "$*"
}

dedup_int_array() {
	# usage: dedup_int_array arrname
	local -n _arr="$1"
	declare -A _seen=()
	local _tmp=() x
	for x in "${!_arr[@]}"; do :; done # quiet shellcheck
	for x in "${_arr[@]}"; do
		[[ -z "$x" ]] && continue
		[[ "$x" =~ ^-?[0-9]+$ ]] || {
			echo "Non-integer in list: '$x'"
			exit 1
		}
		if [[ -z "${_seen[$x]:-}" ]]; then
			_seen[$x]=1
			_tmp+=("$x")
		fi
	done
	mapfile -t _tmp < <(printf '%s\n' "${_tmp[@]}" | sort -n)
	_arr=("${_tmp[@]}")
}

expand_gpu_spec() {
	# usage: expand_gpu_spec "spec" out_array_name
	local spec="$1"
	local -n out="$2"
	out=()
	spec="${spec//[[:space:]]/}" # strip spaces
	if [[ "$spec" =~ ^\[(\-?[0-9]+),(\-?[0-9]+)(\)|\])$ ]]; then
		local start="${BASH_REMATCH[1]}"
		local end="${BASH_REMATCH[2]}"
		local close="${BASH_REMATCH[3]}"
		if [[ "$close" == ")" ]]; then end=$((end - 1)); fi
		((start <= end)) || {
			echo "Invalid GPU range: $spec"
			exit 1
		}
		local i
		for ((i = start; i <= end; i++)); do out+=("$i"); done
	else
		IFS=',' read -r -a out <<<"$spec"
	fi
	dedup_int_array out
}

detect_all_gpus_count() {
	require nvidia-smi
	# Count device indices; use query for machine-parsable output.
	# nvidia-smi --query-gpu=index --format=csv,noheader
	nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | xargs
}

complement_gpus() {
	# usage: complement_gpus total used_array_name out_array_name
	local total="$1"
	local -n used="$2"
	local -n out="$3"
	declare -A usedmap=()
	local u i
	for u in "${used[@]}"; do usedmap["$u"]=1; done
	out=()
	for ((i = 0; i < total; i++)); do
		[[ -z "${usedmap[$i]:-}" ]] && out+=("$i")
	done
}

allocate_ports_seq() {
	# usage: allocate_ports_seq base count out_array_name
	local base="$1" count="$2"
	local -n out="$3"
	out=()
	local i
	for ((i = 0; i < count; i++)); do out+=($((base + i))); done
}

parse_ports_csv() {
	# usage: parse_ports_csv "7100,7101" out_array_name
	local csv="$1"
	local -n out="$2"
	IFS=',' read -r -a out <<<"$csv"
}

check_num_gpus() {
	local have need
	have=$(detect_all_gpus_count)
	need=$((${#PGPUS[@]} + ${#DGPUS[@]}))
	if ((have < need)); then
		echo "GPUs needed=$need, found=$have"
		exit 1
	fi
}

ensure_runtime() {
	require python3
	require curl
	require vllm
	pyreq lmcache
	pyreq fastapi
	pyreq uvicorn
	pyreq httpx
	pyreq nixl
}

cleanup() {
	echo "Stopping all…"
	trap - INT TERM
	pkill -9 -f "disagg_proxy_server.py" || true
	kill -- -$$ 2>/dev/null || true
	wait || true
	exit 0
}
trap cleanup INT TERM

wait_for_server() {
	local port=$1
	local start=$(date +%s)
	echo "Waiting for :$port …"
	until curl -sSf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do
		if (($(date +%s) - start >= TIMEOUT_SECONDS)); then
			echo "Timeout waiting on :$port"
			return 1
		fi
		sleep 1
	done
	echo "Port :$port ready."
}

# ---------- Inline YAML (heredocs) ----------
generate_yaml() {
	mkdir -p configs

	# Shared prefiller config
	cat >configs/lmcache-prefiller-config.yaml <<EOF
# Auto-generated by disagg_example_xpyd_lmcache.sh
local_cpu: False
max_local_cpu_size: 0
max_local_disk_size: 0

enable_nixl: True
enable_xpyd: True
nixl_role: "sender"
nixl_proxy_host: "localhost"
nixl_proxy_port: ${NIXL_PROXY_PORT}
nixl_buffer_size: 1019215872   # 972 MiB-ish
nixl_buffer_device: "cuda"
EOF

	# Per-decoder configs: peer init/alloc ports vary by index
	for i in "${!DGPUS[@]}"; do
		local_init=$((NIXL_PEER_INIT_BASE + i))
		local_alloc=$((NIXL_PEER_ALLOC_BASE + i))
		cat >"configs/lmcache-decoder-$((i + 1))-config.yaml" <<EOF
# Auto-generated by disagg_example_xpyd_lmcache.sh
local_cpu: False
max_local_cpu_size: 0

enable_nixl: True
enable_xpyd: True
nixl_role: "receiver"
nixl_peer_host: "localhost"
nixl_peer_init_port: ${local_init}
nixl_peer_alloc_port: ${local_alloc}
nixl_buffer_size: 2038431744   # ~1.9 GiB
nixl_buffer_device: "cuda"
nixl_backends: [UCX]
EOF
	done
}

# ---------- Launch ----------
main() {
	check_required_files
	check_hf_token
	ensure_runtime

	# Resolve GPU lists ----------------------------------------------------------
	RAW_PREFILL_SPEC="${PREFILL_GPUS:-$DEFAULT_PREFILL_GPUS}"
	RAW_DECODE_SPEC="${DECODE_GPUS:-}"

	expand_gpu_spec "$RAW_PREFILL_SPEC" PGPUS
	if [[ -n "$RAW_DECODE_SPEC" ]]; then
		expand_gpu_spec "$RAW_DECODE_SPEC" DGPUS
	else
		if ((USER_SET_PREFILL == 1 && USER_SET_DECODE == 0)); then
			total=$(detect_all_gpus_count)
			complement_gpus "$total" PGPUS DGPUS
		else
			expand_gpu_spec "$DEFAULT_DECODE_GPUS" DGPUS
		fi
	fi

	# Resolve ports --------------------------------------------------------------
	if ((USER_SET_PPORTS == 1)); then
		parse_ports_csv "${PREFILL_PORTS}" PPORTS
	else
		allocate_ports_seq "$BASE_PREFILL_PORT" "${#PGPUS[@]}" PPORTS
	fi

	if ((USER_SET_DPORTS == 1)); then
		parse_ports_csv "${DECODE_PORTS}" DPORTS
	else
		allocate_ports_seq "$BASE_DECODE_PORT" "${#DGPUS[@]}" DPORTS
	fi

	# Sanity
	if ((${#PGPUS[@]} != ${#PPORTS[@]})); then
		echo "Prefill GPU/port count mismatch"
		exit 1
	fi
	if ((${#DGPUS[@]} != ${#DPORTS[@]})); then
		echo "Decode GPU/port count mismatch"
		exit 1
	fi

	check_num_gpus

	# Pretty print config
	prefill_gpus_str=$(join_by , "${PGPUS[@]}")
	prefill_ports_str=$(join_by , "${PPORTS[@]}")
	decode_gpus_str=$(join_by , "${DGPUS[@]}")
	decode_ports_str=$(join_by , "${DPORTS[@]}")

	echo -e "LMCache XpYd: model=$MODEL\nPREFILL_GPUS=$prefill_gpus_str on $prefill_ports_str\nDECODE_GPUS=$decode_gpus_str on $decode_ports_str\nProxy=$PROXY_PORT  NIXL: proxy=$NIXL_PROXY_PORT, peer-init=$NIXL_PEER_INIT_BASE, peer-alloc=$NIXL_PEER_ALLOC_BASE\nUCX_TLS=$UCX_TLS\n"

	generate_yaml

	echo "Launching proxy on :$PROXY_PORT …"
	PROXY_ARGS=(--host 0.0.0.0 --port "$PROXY_PORT" --prefiller-host 127.0.0.1 --decoder-host 127.0.0.1)
	for p in "${PPORTS[@]}"; do PROXY_ARGS+=(--prefiller-port "$p"); done
	for p in "${DPORTS[@]}"; do PROXY_ARGS+=(--decoder-port "$p"); done
	python3 disagg_proxy_server.py "${PROXY_ARGS[@]}" >proxy.log 2>&1 &
	PIDS+=($!)

	echo "Starting ${#PGPUS[@]} prefiller(s)…"
	for i in "${!PGPUS[@]}"; do
		gpu=${PGPUS[$i]}
		port=${PPORTS[$i]}
		rpc="producer$((i + 1))"
		echo "  Prefiller $((i + 1)): GPU ${gpu} → :${port} (rpc=${rpc})"
		UCX_TLS="$UCX_TLS" \
			LMCACHE_CONFIG_FILE="configs/lmcache-prefiller-config.yaml" \
			CUDA_VISIBLE_DEVICES="${gpu}" \
			vllm serve "$MODEL" \
			--host 0.0.0.0 \
			--port "$port" \
			--no-enable-prefix-caching \
			--tensor-parallel-size 1 \
			--seed 1024 \
			--max-model-len 10000 \
			--max-num-batched-tokens 10000 \
			--max-num-seqs 256 \
			--gpu-memory-utilization 0.7 \
			--kv-transfer-config \
			"{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"discard_partial_chunks\":false,\"lmcache_rpc_port\":\"${rpc}\"}}" \
			>"prefiller$((i + 1)).log" 2>&1 &
		PIDS+=($!)
	done

	echo "Starting ${#DGPUS[@]} decoder(s)…"
	for i in "${!DGPUS[@]}"; do
		gpu=${DGPUS[$i]}
		port=${DPORTS[$i]}
		rpc="consumer$((i + 1))"
		cfg="configs/lmcache-decoder-$((i + 1))-config.yaml"
		echo "  Decoder $((i + 1)): GPU ${gpu} → :${port} (rpc=${rpc})"
		UCX_TLS="$UCX_TLS" \
			LMCACHE_CONFIG_FILE="$cfg" \
			CUDA_VISIBLE_DEVICES="${gpu}" \
			vllm serve "$MODEL" \
			--host 0.0.0.0 \
			--port "$port" \
			--disable-log-requests \
			--no-enable-prefix-caching \
			--tensor-parallel-size 1 \
			--seed 1024 \
			--max-model-len 10000 \
			--max-num-batched-tokens 10000 \
			--max-num-seqs 256 \
			--gpu-memory-utilization 0.7 \
			--kv-transfer-config \
			"{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"discard_partial_chunks\":false,\"lmcache_rpc_port\":\"${rpc}\",\"skip_last_n_tokens\":1}}" \
			>"decoder$((i + 1)).log" 2>&1 &
		PIDS+=($!)
	done

	echo "Waiting for vLLM servers…"
	for p in "${PPORTS[@]}" "${DPORTS[@]}"; do
		wait_for_server "$p" || {
			cleanup
			exit 1
		}
	done

	echo "Waiting for proxy…"
	wait_for_server "$PROXY_PORT" || {
		cleanup
		exit 1
	}

	echo "All services up on :$PROXY_PORT. Press Ctrl-C to tear down."
	wait
}

main "$@"
