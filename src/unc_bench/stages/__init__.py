"""The five pipeline stages.

Each stage reads parquet, writes parquet, and is safe to rerun. Nothing here
imports torch at module level, so the CLI stays importable in CI where neither
torch nor transformers is installed.
"""
