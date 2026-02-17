# `scripts/` — Operational & Demo Scripts

Scripts for verification, demos, deployment testing, and cost estimation. All scripts are designed to be run from the project root against a running API instance.

## Quick Reference

### Verification (CI/Review)
| Script | Purpose | When to use |
|--------|---------|-------------|
| `run_full_ci_suite.py` | Runs Pyright + Pytest + Docker build in sequence | Before any commit or PR |
| `verify_ecosystem.py` | End-to-end system verification (CI, stress, drift, costs) | Final pre-release check |
| `verify_fresh_install.py` | Validates a clean environment has all dependencies | First-time setup |
| `verify_resilience.py` | Concurrency stress, large payloads, malformed data | Testing API robustness |

### Demos (Live Presentation)
| Script | Purpose | Expected Result |
|--------|---------|-----------------|
| `demo_classify.py` | Classification accuracy on diverse samples | ~73% on short-text queries |
| `demo_ood.py` | OOD rejection on nonsense/ambiguous inputs | Recall > 80% |
| `demo_stress.py` | Latency under concurrent load | p99 < 200ms |
| `demo_security.py` | Injection and sanitization handling | All attacks handled gracefully |
| `demo_metrics.py` | Telemetry endpoint walkthrough | Shows latency, cache, throughput |

### Analysis & Testing
| Script | Purpose |
|--------|---------|
| `reproduce_board_metrics.py` | Reproduces exact F1 scores from the board report using test data |
| `analyze_threshold_sensitivity.py` | Sweeps OOD threshold values and plots precision/recall tradeoff |
| `estimate_scaling_costs.py` | Benchmarks latency/throughput and projects AWS Lambda vs EC2 costs |
| `simulate_data_drift.py` | Simulates OOD traffic drift and monitors metric degradation |
| `test_deployment.py` | Tests a remote deployment (Railway) via CLI args |
| `test_integration.py` | Basic integration test (health + classify) |
| `test_health.py` | Minimal health check |
| `run_api.py` | Legacy API launcher (use `uvicorn app.main:app` instead) |

## Design Decisions

- **All scripts use HTTP**: They test the API as a black box via `requests`/`httpx`, not by importing app internals. This means the same scripts work against local, Docker, or Railway deployments.
- **`test_deployment.py` uses CLI args, not `.env`**: Avoids config leakage. Prints a step-by-step guide if args aren't provided.
- **`requests.Session()`**: Used in benchmarks for TCP connection reuse, avoiding Windows-specific connection overhead (~2s per new connection).
