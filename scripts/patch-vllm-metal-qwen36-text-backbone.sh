#!/usr/bin/env bash
set -euo pipefail

VLLM_PYTHON="${VLLM_METAL_PYTHON:-${HOME}/.venv-vllm-metal/bin/python}"

if [ ! -x "${VLLM_PYTHON}" ]; then
	printf 'vLLM-Metal Python not found: %s\n' "${VLLM_PYTHON}" >&2
	printf 'Set VLLM_METAL_PYTHON or install vLLM-Metal first.\n' >&2
	exit 127
fi

"${VLLM_PYTHON}" - <<'PY'
from __future__ import annotations

from pathlib import Path
import py_compile

import vllm_metal.v1.model_adapter as model_adapter

path = Path(model_adapter.__file__)
text = path.read_text()

old = '''        if multimodal_mode == "text-only-compat":
            return self._matches_auto_text_backbone_override(hf_config)
        return self._matches_auto_text_backbone_override(hf_config)
'''

new = '''        if multimodal_mode == "text-only-compat":
            if self._matches_auto_text_backbone_override(hf_config):
                return True
            architectures = getattr(hf_config, "architectures", ()) or ()
            return any(
                arch in _TEXT_BACKBONE_OVERRIDE_ARCHITECTURES
                for arch in architectures
            )
        return self._matches_auto_text_backbone_override(hf_config)
'''

if new in text:
    print(f"already patched: {path}")
elif old in text:
    path.write_text(text.replace(old, new))
    py_compile.compile(str(path), doraise=True)
    print(f"patched: {path}")
else:
    raise SystemExit(
        "unsupported vLLM-Metal model_adapter.py shape; inspect manually: "
        f"{path}"
    )
PY
