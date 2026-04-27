# Evidence for issue #4 — V_h structural inversion

Plots and analysis referenced from [issue #4](https://github.com/ajaytalati/Python-Model-OT-Control/issues/4). All scans use the SWAT recovery scenario with default parameters from `_vendored_models/swat/parameters.py`.

## Plots (Phase 1.5 reconnaissance)

| File | What it shows |
|---|---|
| `coarse_vh_vn_scan.png` | Coarse 6×4 V_h × V_n grid, T_0=0.05, D=60 days. Mean T(D), mean E (over horizon), mean E (terminal). Maximum T(D) is at V_h=0. |
| `fine_scan.png` | Fine 11×11 V_h × V_n grid, T_0=0.55, D=60 days. Resolves the maximum to V_h=0, V_n≈0.4. |
| `deterministic_comparison.png` | Same scan under three noise regimes (stochastic / Stuart-Landau deterministic / fully deterministic). The structural inversion is identical across all three — proves the inversion is in the drift, not the diffusion. |
| `diagnostic.png` | T(t) and E(t) under three constant control vectors (reference / "idealised healthy" v1.2.0 / bounds-max), starting from both T_0=0.05 and T_0=0.55. |

## Analysis docs

| File | Contents |
|---|---|
| `proof_v_h_optimum_at_zero.md` | Formal mathematical proof, from first principles, that under the SWAT drift formula $\mathrm{amp}_W$ is bell-shaped in $V_h$ with peak at $V_h^\star = -(V_n - a + \alpha_T T)$, and that this peak is constrained to the boundary $V_h = 0$ across the entire clinically-relevant region. Pencil-and-paper, no simulation. |
| `model_fix_plan.md` | Three candidate fixes (Option A: direct anabolic coupling on T — recommended; Option B: sign flip on V_h in u_W; Option C: V_h as circadian-amplitude modulator) with concrete code-edit lists, calibration plan for $\alpha_h$, and downstream re-vendor steps. |
| `upstream_gating_tests.md` | Nine pytest-style tests that the upstream model must pass before vendoring. Five of them would have caught issue #4 on the original release. Includes suggested CI integration. |
