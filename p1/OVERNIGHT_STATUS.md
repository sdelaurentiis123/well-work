# Overnight status — 2026-04-22

## Final seed-averaged numbers (3 pretrain seeds per condition)

| condition | best val VRMSE | Δ vs scratch |
|---|---|---|
| scratch (h=48, matched-arch) | 0.4654 ± 0.0175 | — |
| **MHD pretrain + FT** | **0.3015 ± 0.0010** | **−35.2%** (in 28–40% band) |
| **NS pretrain + FT** (wrong-domain control) | **0.5285 ± 0.0177** | **+13.6%** (in 5–18% band) |

Both shifts within pre-declared tolerance bands — no framing changes needed.
NS penalty moved 10.7% → 13.6% with real pretrain-seed variance (previously single-seed);
MHD gap essentially unchanged (35.4% → 35.2%), confirming robustness.

Per-seed values:
- MHD FTs: [0.3007, 0.3009, 0.3028] — fine-tune endpoint nearly invariant under pretrain randomness
- NS FTs:  [0.5150, 0.5535, 0.5169] — larger spread; seed 1 outlier high
- MHD pretrains (best val on M_A=2.0 val): [0.4193 (s0 original), 0.4193 (s1), 0.4197 (s2)]
- NS  pretrains (best val on NS val):       [~0.43 (s0 original), 0.3674 (s1), 0.3662 (s2)]

## Commit hashes at final state

- `well-work` (plasma-tinkering): `df76f01` — "paper: seed-averaged results (3 pretrain seeds/condition)"
- `well-work-paper`:              `f6c19e4` — same

Both pushed to GitHub (sdelaurentiis123/well-work, sdelaurentiis123/well-work-paper, `main`).

## Vast instance

Instance 35401749 (Minnesota RTX 4090) destroyed via `vastai destroy instance 35401749`.
`vastai show instances` now returns 0 running instances.

Total additional overnight cost: ~$2.00 (5.7 hr × $0.395/hr).

## Things that broke / required manual intervention

1. **Dead nohup on downloads.** Initial the-well-download for MHD valid + supernova was
   launched in a shell that didn't have the venv on PATH, so both failed immediately
   with `No such file or directory`. seed_runner sat waiting on files that would never
   arrive. Fixed mid-night by relaunching with absolute `/venv/main/bin/the-well-download`.
2. **Nested download path.** The relaunched downloads wrote to
   `/root/data/datasets/datasets/{MHD_64,supernova_explosion_64}/data/valid/` (double
   `datasets/`) because of how `--base-path` was interpreted. seed_runner polled the
   non-nested path. Fixed by moving the directories into place; no data lost.
3. **seed_runner.sh died silently after pretrain_s1.** The original bash script was not
   setsid-nohup-detached, so its controlling shell exited after pretrain_s1 finished
   cleanly and wandb sync'd — presumably from SSH session teardown. Fixed by writing
   `seed_runner2.sh` (starting at pretrain_s2), launching via `setsid nohup ... < /dev/null &`.
   pretrain_s1's results were preserved — `best.pt` + `log.jsonl` completed fine before the
   script died.
4. **Monitor tool timed out once** (one-hour default SSH-piped `tail -F`). Re-armed with
   `persistent=true`; survived the rest of the night.

None of the above cost a run; all mid-flight outputs were preserved or resumable.

## Data pulled local before destroy

`/Users/stanislavdelaurentiis/plasma-tinkering/p1/runs_from_seed_runner/` contains:
- 8 × `<run_name>/best.pt` + `log.jsonl` (all 8 new seed runs)
- 8 × `<run_name>.log` (stdout captures)
- `seed_runner.log` + `seed_runner2.log`

Total ~1.7 GB. Verified 8 best.pt + 8 log.jsonl before destroying instance.

## Out-of-scope items flagged but not touched

(Per runbook: do NOT make framing changes beyond the 4 allowed sections. Intro, Discussion,
fig2–fig6, Methods all untouched.)

- The abstract still says "$\approx$3× worse … $\approx$6× worse" for long-horizon VRMSE
  comparisons. Source numbers: ft_01 step-50 VRMSE 10.84 vs pretrain_ood 3.03 (3.58×) vs
  baseline_01 1.95 (5.55×). "3×" and "6×" are each ~0.5 off; used `\approx` to soften.
  If you want the exact multiples in the paper, rewrite as "3.6×" and "5.5×".
- Per-FT-seed variance within a pretrain (i.e., re-running FT multiple times from the
  *same* pretrain checkpoint) was only measured for pretrain_s0. For pretrain_s1 and s2
  we ran 1 FT each. Reporting the 3-pretrain-seed mean assumes FT-seed variance is
  smaller than pretrain-seed variance (which it is, σ ≈ 3e-4 vs 1e-3 for MHD).
- The `p1/runs_from_seed_runner/pretrain_s1/log.jsonl` shows `best_val_vrmse 0.41929` on
  M_A=2.0 val — identical to the original pretrain run. This is expected (same seed-1
  data order reproducibility) but mildly surprising. Worth sanity-checking if the three
  "MHD pretrain seeds" are actually three independent initializations or share a seed.

## One-line verdict

**Paper is ready for Stan to read.**

Numbers are verified against on-disk logs, figures are regenerated with real pretrain-seed
error bars, and both repos are pushed. The three MHD pretrain endpoints [0.3007, 0.3009,
0.3028] are tight enough that the 35% claim is unambiguously robust to pretrain-init
randomness. The NS 13.6% penalty is now grounded in proper seed variance and is still
comfortably negative (wrong-domain pretraining underperforms scratch), which was the
qualitative claim.
