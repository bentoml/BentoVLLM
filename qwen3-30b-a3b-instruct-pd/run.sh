#!/usr/bin/env bash
set -x

kubectl run sgl-dev \
  --image=docker.io/rocm/sgl-dev:v0.5.3rc0-rocm630-mi30x-20251002 \
  --restart=Never \
  --overrides='{
  "spec": {
    "hostNetwork": true,
    "hostIPC": true,
    "containers": [{
      "name": "sgl-dev",
      "image": "docker.io/rocm/sgl-dev:v0.5.3-rocm700-mi30x-20251008",
      "command": ["/bin/bash"],
      "args": ["-c", "sleep infinity"],
      "env": [
        {
          "name": "CUDA_VISIBLE_DEVICES",
          "value": "0,1,2,3,4,5,6,7"
        },
        {
          "name": "HF_HUB_CACHE_DIR",
          "value": "/root/.cache/huggingface/hub"
        }
      ],
      "securityContext": {
        "privileged": true,
        "capabilities": {
          "add": ["SYS_ADMIN", "SYS_PTRACE"]
        },
        "seccompProfile": {
          "type": "Unconfined"
        }
      },
      "volumeMounts": [
        {
          "name": "workspace",
          "mountPath": "/workspace"
        },
        {
          "name": "hf-cache",
          "mountPath": "/root/.cache"
        },
        {
          "name": "dshm",
          "mountPath": "/dev/shm"
        },
        {
          "name": "dev-kfd",
          "mountPath": "/dev/kfd"
        },
        {
          "name": "dev-dri",
          "mountPath": "/dev/dri"
        }
      ],
      "resources": {
        "limits": {
          "amd.com/gpu": 8
        }
      }
    }],
    "volumes": [
      {
        "name": "workspace",
        "hostPath": {
          "path": "/root/workspace/amd-staging/sglang-r1-pd",
          "type": "Directory"
        }
      },
      {
        "name": "hf-cache",
        "hostPath": {
          "path": "/root/.cache",
          "type": "Directory"
        }
      },
      {
        "name": "dshm",
        "emptyDir": {
          "medium": "Memory",
          "sizeLimit": "32Gi"
        }
      },
      {
        "name": "dev-kfd",
        "hostPath": {
          "path": "/dev/kfd"
        }
      },
      {
        "name": "dev-dri",
        "hostPath": {
          "path": "/dev/dri"
        }
      }
    ]
  }
}'
