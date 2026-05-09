## v0.2.0 — April 2026

### Added
- GymnasiumAdapter: wraps any Gymnasium-compatible environment
- StableBaselines3Adapter: wraps trained SB3 models with as_agent_fn()
- CustomAdapter: base class for any custom environment
- RLlibAdapter: wraps trained Ray RLlib algorithms with as_agent_fn()

### Changed
- Package description updated to reflect universal compatibility
- Optional dependencies restructured by ecosystem
- Version bumped from 0.1.6 to 0.2.0