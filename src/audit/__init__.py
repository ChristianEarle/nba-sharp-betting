"""Audit-only package (read-heavy scrutiny tooling) — never imported by production
board/pipeline code, and never writes to a production-path artifact. See
``src.audit.holdout_board_audit`` for the 2024/2025 holdout board simulation this
package exists for.
"""
