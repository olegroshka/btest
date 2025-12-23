"""Local-first platform API.

This package provides an HTTP API for:
- listing available data providers/datasets/snapshots
- exploring cached ArcticDB data
- triggering data materialization (tail-fill)

It is intentionally local-first (single-user) but designed to evolve into a
multi-user service with a job queue.
"""

