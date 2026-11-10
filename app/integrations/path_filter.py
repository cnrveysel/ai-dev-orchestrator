"""Classification of repository paths for change attribution and review context.

Applied by the orchestrator itself, regardless of the repository's .gitignore:

  generated  dependency / build output / cache / compiled files. Never attributed
             to the worker, never snapshotted, never sent to the reviewer.
             (Nothing is ever deleted.)
  metadata   authored but not worth diffing: lockfiles and binary assets.
             Listed as changed, diff omitted.
  diffable   source / config / docs. Eligible for the review diff.
"""
from __future__ import annotations

import posixpath
from collections import Counter

GENERATED = "generated"
METADATA = "metadata"
DIFFABLE = "diffable"

# Directory names that are generated wherever they appear (compared case-insensitively).
GENERATED_DIRS = (
    "node_modules", "dist", "build", "coverage", ".vite", ".next", ".cache", ".venv", "venv",
    "__pycache__", "Binaries", "Intermediate", "Saved", "DerivedDataCache",
    # temporary verification artifacts (the V2 worker is told to put smoke files here) and caches
    "scratch_smoke", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".parcel-cache", ".turbo",
    ".nyc_output", ".gradle", ".tmp", "tmp", "temp",
)
# Obvious generated / cache / compiled files and temporary bundles.
GENERATED_FILE_GLOBS = (
    "*.pyc", "*.pyo", "*.pyd", "*.o", "*.lib", "*.a", "*.so", "*.dll", "*.dylib", "*.exe",
    "*.pdb", "*.ilk", "*.exp", "*.class", "*.jar", "*.wasm", "*.map", "*.min.js", "*.min.css",
    "*.tsbuildinfo", "*.log", ".DS_Store", "Thumbs.db", "desktop.ini",
    "*.tmp", "*.temp", "*.bundle.js", "*.bundle.mjs", "*.bundle.cjs",
)
LOCKFILES = frozenset(n.lower() for n in (
    "package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb", "poetry.lock",
    "Pipfile.lock", "Cargo.lock", "composer.lock", "Gemfile.lock", "uv.lock",
))
BINARY_ASSET_EXTS = frozenset((
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".tga", ".psd", ".webp", ".dds", ".exr", ".hdr",
    ".wav", ".mp3", ".ogg", ".flac", ".mp4", ".mov", ".webm", ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".zip", ".7z", ".rar", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".uasset", ".umap", ".upk", ".pak",
    ".ubulk", ".uexp", ".fbx", ".obj", ".blend", ".glb", ".pdf",
))
DIFFABLE_EXTS = frozenset((
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".cs",
    ".html", ".css", ".scss", ".json", ".toml", ".yaml", ".yml", ".md", ".txt", ".ini", ".cfg",
    ".sh", ".ps1", ".uproject", ".uplugin",
))
DIFFABLE_NAMES = frozenset((".gitignore", ".gitattributes", ".editorconfig", "dockerfile", "makefile"))

_GEN_DIRS_LOWER = frozenset(d.lower() for d in GENERATED_DIRS)


def _parts(path: str) -> list[str]:
    return [p for p in path.replace("\\", "/").split("/") if p]


def generated_dir(path: str) -> str | None:
    """The generated directory component of `path` (e.g. 'node_modules'), if any."""
    for part in _parts(path)[:-1]:
        if part.lower() in _GEN_DIRS_LOWER:
            return part
    return None


def _matches_generated_file(name: str) -> bool:
    low = name.lower()
    for g in GENERATED_FILE_GLOBS:
        g = g.lower()
        if g.startswith("*") and low.endswith(g[1:]):
            return True
        if low == g:
            return True
    return False


def classify(path: str) -> str:
    parts = _parts(path)
    if not parts:
        return GENERATED
    name = parts[-1]
    if generated_dir(path) or _matches_generated_file(name):
        return GENERATED
    low = name.lower()
    if low in LOCKFILES:
        return METADATA
    ext = posixpath.splitext(low)[1]
    if ext in BINARY_ASSET_EXTS:
        return METADATA
    if ext in DIFFABLE_EXTS or low in DIFFABLE_NAMES:
        return DIFFABLE
    return METADATA  # unknown kinds are listed but never embedded


def is_generated(path: str) -> bool:
    return classify(path) == GENERATED


def metadata_reason(path: str) -> str:
    low = _parts(path)[-1].lower() if _parts(path) else ""
    if low in LOCKFILES:
        return "lockfile"
    if posixpath.splitext(low)[1] in BINARY_ASSET_EXTS:
        return "binary asset"
    return "not a source/config/doc type"


def summarize_generated(paths) -> Counter:
    """Count generated paths by their generated directory (or 'generated files')."""
    c: Counter = Counter()
    for p in paths:
        c[generated_dir(p) or "generated files"] += 1
    return c


def git_exclude_pathspecs() -> list[str]:
    """Pathspecs that exclude generated paths from git status/add/diff (any depth, any case)."""
    specs = [f":(exclude,glob,icase)**/{d}/**" for d in GENERATED_DIRS]
    specs += [f":(exclude,glob,icase)**/{g}" for g in GENERATED_FILE_GLOBS]
    return specs


def git_lockfile_exclude_pathspecs() -> list[str]:
    return [f":(exclude,glob,icase)**/{n}" for n in sorted(LOCKFILES)]
