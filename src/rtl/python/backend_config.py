#!/usr/bin/env python3
"""Shared backend configuration loader for RTL EDA tools.

Reads configs/backend/config.json to resolve tool paths and library locations.
Falls back to environment variables and environment-modules when config is
absent or incomplete.
"""

import json
import os
import shutil
import subprocess

_config_cache = None


def _find_repo_root():
    """Walk up from this file looking for configs/backend directory."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, "configs", "backend")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def load_config():
    """Load configs/backend/config.json, returning a dict (empty on failure)."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    root = _find_repo_root()
    if root is None:
        _config_cache = {}
        return _config_cache

    cfg_path = os.path.join(root, "configs", "backend", "config.json")
    if not os.path.isfile(cfg_path):
        _config_cache = {}
        return _config_cache

    try:
        with open(cfg_path) as f:
            _config_cache = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[backend_config] WARNING: failed to load {cfg_path}: {e}")
        _config_cache = {}
    return _config_cache


def _env_modules_hint():
    """Return a hint string about environment-modules if available."""
    if os.path.isfile("/etc/profile.d/modules.sh"):
        return ("  Hint: environment-modules detected. Try:\n"
                "    source /etc/profile.d/modules.sh && module avail\n"
                "  to see available tool modules.")
    return ""


def _try_module_load(name, module_spec):
    """Attempt to load an environment module and locate the tool binary."""
    try:
        result = subprocess.run(
            ["bash", "-c",
             f"source /etc/profile.d/modules.sh && "
             f"module load {module_spec} && which {name}"],
            capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def resolve_tool(name, fallback_module=None):
    """Resolve an EDA tool executable path.

    Priority:
      1. config.json tools.<name>.path (explicit path)
      2. shutil.which (already on PATH)
      3. module load (from config or fallback_module)

    Returns the executable path (str) or None if not found.
    """
    cfg = load_config()
    tool_cfg = cfg.get("tools", {}).get(name, {})

    # 1. Explicit path from config
    explicit = tool_cfg.get("path", "")
    if explicit and os.path.isfile(explicit):
        return explicit

    # 2. Already on PATH
    on_path = shutil.which(name)
    if on_path:
        return on_path

    # 3. Module loading
    module_spec = tool_cfg.get("module", "") or fallback_module
    if module_spec:
        loaded = _try_module_load(name, module_spec)
        if loaded:
            return loaded

    hint = _env_modules_hint()
    print(f"[backend_config] '{name}' not found.")
    if hint:
        print(hint)
    return None


def resolve_lib(key, env_var=None):
    """Resolve a library path.

    Priority:
      1. Environment variable (explicit, used in CI)
      2. config.json value at the dotted key path (e.g. "synth.lib_search_path")

    Returns the path string or empty string if not found.
    """
    # 1. Environment variable
    if env_var:
        val = os.environ.get(env_var, "")
        if val:
            return val

    # 2. Config file
    cfg = load_config()
    parts = key.split(".")
    node = cfg
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p, "")
        else:
            return ""
    return node if isinstance(node, str) else ""


def reset_cache():
    """Clear cached config (useful for testing)."""
    global _config_cache
    _config_cache = None
