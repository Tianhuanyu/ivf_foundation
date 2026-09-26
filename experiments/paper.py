#!/usr/bin/env python3
"""paper.py — the paper's experiment plan as code: claims -> experiments -> commands -> tables.

  python experiments/paper.py status              # what's done / missing for every experiment
  python experiments/paper.py commands [E1 E2..]  # exact commands (in order) for what's still missing
  python experiments/paper.py tokens              # C3 analysis: object size in ViT tokens (cheap, local)
  python experiments/paper.py tables              # paper tables (markdown) -> experiments/out/

Nothing here launches training or GCP by itself: `commands` prints what to run (GCP costs money and
needs a human go-ahead). The single source of truth for each piece stays where it was:
  DAPT arms / budgets ........ scripts/dapt_arms.sh           (training repo)
  benchmark matrix / hparams . stage1_out/benchmark/runner.py (benchmark repo)
  locked splits .............. stage1_out/benchmark/dataset/split_lists/
  result loading / seeds ..... stage1_out/benchmark/report_common.py
The narrative (claims, decision rules) is in PAPER.md; keep the two in sync.
"""
import argparse
import hashlib
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

DT = Path(__file__).resolve().parents[1]
_default_bench = r"D:\Conceivable-SharedData01-23Jun2026" if os.name == "nt" else "/mnt/d/Conceivable-SharedData01-23Jun2026"
BENCH = Path(os.environ.get("BENCH_ROOT", _default_bench))
sys.path.insert(0, str(BENCH / "stage1_out"))
from benchmark import report_common as rc  # noqa: E402
from benchmark import runner  # noqa: E402

OUT = DT / "experiments" / "out"

# ── the plan ─────────────────────────────────────────────────────────────────────────────────
BUDGET = "long"                                            # every DAPT arm gets the SAME budget
MAIN_ARM = "motion_weighted"                               # "ours"
ABLATION_ARMS = ["uniform", "content", "motion", "motion_weighted"]
APPENDIX_ARMS = ["motion_ibotlocal_high", "motion_ibotlocal_low"]
MAIN_BACKBONES = ["resnet50_fpn", "dinov2_s_fpn", "dinov2_b_fpn", "biomedclip_fpn", "dinov3_b_fpn", "dinov3_dapt_b_fpn"]
# main benchmark runs detection at 1024 px (contract amendment A2); E4 re-runs it at 320 px
RES_ABL = dict(imgsz=320, protocol="frozen",
               backbones=["resnet50_fpn", "dinov2_b_fpn", "dinov3_b_fpn", "dinov3_dapt_b_fpn"],
               datasets=["holding_pip", "routine2_coc", "cvit_incubator", "cvit_workstation"])
SEEDS = runner.SEEDS
ALL_DATASETS = list(runner.TASK_CFG)                       # the runner always runs all 9 (DROP ones are reported excluded)
DETECT_DATASETS = [d for d in ALL_DATASETS if d not in ("cellasp", "icsi_seg")]
RES_ABL_DIR = BENCH / "stage1_out" / f"benchmark_results_imgsz{RES_ABL['imgsz']}"
PATCH = 16

# measured cost anchors (A100-40GB): e1 DAPT 3000 it @ batch16 = 51 min (1.0 s/it, gcp_outputs log 2026-09-22);
# main benchmark 108 runs = 8.9 h (~5 min/run at 224/320 px, gcp_bench_outputs/bench_run.log).
# 1024-px detection runs are slower (not yet measured) -> estimates below are lower bounds.
MIN_PER_BENCH_RUN_320 = 5.0


def arm_weights(arm: str):
    """Where a DAPT arm's backbone lands: local dapt_run.sh -> weights/, GCP fetch -> gcp_outputs/."""
    name = f"dinov3_vitb16_dapt_{arm}_{BUDGET}_backbone.pth"
    for d in (DT / "weights", DT / "gcp_outputs"):
        if (d / name).is_file():
            return d / name
    return None


_sha_cache = {}


def sha256(p: Path) -> str:
    key = (str(p), p.stat().st_size, p.stat().st_mtime_ns)
    if key not in _sha_cache:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        _sha_cache[key] = h.hexdigest()
    return _sha_cache[key]


def runs_in(results_dir: Path):
    return rc.load_runs(results_dir, include_drop=True, quiet=True) if results_dir.is_dir() else []


def count(runs, backbone, sha=None, protocols=None, datasets=None):
    keep = [r for r in runs if r.backbone == backbone and (sha is None or r.weights_sha256 == sha)
            and (protocols is None or r.protocol in protocols) and (datasets is None or r.dataset in datasets)
            and r.seed in SEEDS]
    return len({(r.dataset, r.protocol, r.seed) for r in keep})


def to_posix(p: Path) -> str:
    s = str(p).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}" if len(s) > 1 and s[1] == ":" else s


# ── experiments ──────────────────────────────────────────────────────────────────────────────
def exp_status():
    """-> list of (id, title, done, total, [commands for what's missing], note)"""
    main_runs = runs_in(runner.OUTDIR)
    abl_runs = runs_in(RES_ABL_DIR)
    rows = []

    # E1 — DAPT pretraining of every arm at the same budget
    missing = [a for a in ABLATION_ARMS if arm_weights(a) is None]
    cmds = []
    for a in missing:
        cmds += [f"# {a}: ~16-17 h on A100 (estimate: 20000 it @ batch48; e1 measured 1.0 s/it @ batch16)",
                 f"ARM={a} BUDGET={BUDGET} ./scripts/dapt_train.sh start && ./scripts/dapt_train.sh status",
                 "./scripts/dapt_train.sh finish      # -> gcp_outputs/dinov3_vitb16_dapt_%s_%s_backbone.pth" % (a, BUDGET)]
    if missing:
        cmds.insert(0, "# prerequisite (once): complete frame cache incl. motion sidecars + frame weights\n"
                       "#   gcloud storage rm gs://mlflow-artifacts-ai-a100/dapt/cache/frames_hires.tar && ./scripts/dapt_prep.sh run")
    rows.append(("E1", f"DAPT pretraining, arms {ABLATION_ARMS} @ BUDGET={BUDGET}",
                 len(ABLATION_ARMS) - len(missing), len(ABLATION_ARMS), cmds, "training repo, WSL"))

    # E2 — main benchmark matrix (claim C1)
    total = len(MAIN_BACKBONES) * len(ALL_DATASETS) * 2 * len(SEEDS)
    done, cmds = 0, []
    main_w = arm_weights(MAIN_ARM)
    for bb in MAIN_BACKBONES:
        if bb == "dinov3_dapt_b_fpn":
            if main_w is None:
                cmds.append(f"# {bb}: waiting for E1 ({MAIN_ARM})")
                continue
            n = count(main_runs, bb, sha256(main_w))
            env = f"DINOV3_DAPT_B_WEIGHTS={to_posix(main_w)} "
        else:
            n, env = count(main_runs, bb), ""
        done += n
        if n < len(ALL_DATASETS) * 2 * len(SEEDS):
            cmds.append(f"{env}python run_benchmark_server.py --all-seeds --backbone {bb} --skip-done")
    rows.append(("E2", "main benchmark: 6 backbones x 9 datasets x 2 protocols x 3 seeds (detect 1024 px)", done, total,
                 [f"cd {to_posix(BENCH)}", "# measure one 1024-px detection run first"] + cmds if cmds else [],
                 f"benchmark repo; >{MIN_PER_BENCH_RUN_320 * (total - done) / 60:.0f} GPU-h left (lower bound, 1024 px not measured)"))

    # E3 — DAPT-arm ablation, frozen (claim C2)
    per_arm = len(ALL_DATASETS) * len(SEEDS)
    done, cmds = 0, []
    for a in ABLATION_ARMS:
        w = arm_weights(a)
        if w is None:
            cmds.append(f"# {a}: waiting for E1")
            continue
        n = count(main_runs, "dinov3_dapt_b_fpn", sha256(w), protocols={"frozen"})
        done += n
        if n < per_arm:
            cmds.append(f"DINOV3_DAPT_B_WEIGHTS={to_posix(w)} python run_benchmark_server.py --all-seeds "
                        f"--backbone dinov3_dapt_b_fpn --protocol frozen --skip-done")
    rows.append(("E3", f"ablation: DAPT arms x 9 datasets x frozen x 3 seeds", done, per_arm * len(ABLATION_ARMS),
                 [f"cd {to_posix(BENCH)}"] + cmds if cmds else [], "main-arm frozen runs are shared with E2"))

    # E4 — resolution (claim C3): the main table's detection runs again at 320 px
    per_bb = len(RES_ABL["datasets"]) * len(SEEDS)
    done, cmds = 0, []
    base = (f"python run_benchmark_hires_ablation.py --all-seeds --imgsz {RES_ABL['imgsz']} "
            f"--protocol {RES_ABL['protocol']} --dataset {' '.join(RES_ABL['datasets'])} --skip-done")
    for bb in RES_ABL["backbones"]:
        if bb == "dinov3_dapt_b_fpn":
            if main_w is None:
                cmds.append(f"# {bb}: waiting for E1 ({MAIN_ARM})")
                continue
            n = count(abl_runs, bb, sha256(main_w), protocols={"frozen"}, datasets=RES_ABL["datasets"])
            env = f"DINOV3_DAPT_B_WEIGHTS={to_posix(main_w)} "
        else:
            n = count(abl_runs, bb, protocols={"frozen"}, datasets=RES_ABL["datasets"])
            env = ""
        done += n
        if n < per_bb:
            cmds.append(f"{env}{base} --backbone {bb}")
    rows.append(("E4", f"resolution ablation: 4 backbones x {len(RES_ABL['datasets'])} detect datasets x frozen x 3 seeds @ {RES_ABL['imgsz']} px",
                 done, per_bb * len(RES_ABL["backbones"]),
                 [f"cd {to_posix(BENCH)}"] + cmds if cmds else [],
                 "+ `paper.py tokens` (local) + verify_v2_perclass_ap.py per run for per-class AP"))

    # E5 — appendix (optional): ibot_local arms
    have = [a for a in APPENDIX_ARMS if arm_weights(a) is not None]
    rows.append(("E5", f"appendix (optional): {APPENDIX_ARMS} @ {BUDGET}, then as E3", len(have), len(APPENDIX_ARMS),
                 [f"ARM={a} BUDGET={BUDGET} ./scripts/dapt_train.sh start" for a in APPENDIX_ARMS if a not in have],
                 "not needed for the main claims"))
    return rows


def cmd_status(_):
    for eid, title, done, total, _, note in exp_status():
        mark = "✅" if done >= total else ("…" if done else "·")
        print(f"{mark} {eid}  {done:>4}/{total:<4} {title}\n            {note}")


def cmd_commands(args):
    for eid, title, done, total, cmds, note in exp_status():
        if args.ids and eid not in args.ids:
            continue
        print(f"\n### {eid} — {title}  [{done}/{total}]")
        print("\n".join(cmds) if cmds else "# complete")


# ── C3 analysis: object size in ViT tokens ──────────────────────────────────────────────────
def cmd_tokens(_):
    from PIL import Image
    from benchmark.dataset.splits import load_split
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Object size in ViT-16 tokens (claim C3)", "",
             "Short / long side of every GT box after letterboxing the image to imgsz (as the benchmark does), "
             f"divided by the ViT patch size ({PATCH} px). All splits of the locked split lists.", "",
             "| dataset | class | boxes | short side @320 (tokens, median) | long side @320 | short @1024 | "
             "boxes < 1 token short side @320 | < 2 tokens @320 |", "|---|---|---|---|---|---|---|---|"]
    for ds in DETECT_DATASETS:
        root = Path(runner.DATA_ROOTS[ds])
        names = {}
        cls_file = root / "classes.txt"
        if cls_file.is_file():
            names = {str(i): n.strip() for i, n in enumerate(cls_file.read_text(encoding="utf-8").splitlines()) if n.strip()}
        by_cls = defaultdict(list)                        # cls -> [(short_px_at_1, long_px_at_1)] as fraction of max side
        for split in ("train", "val", "test"):
            for rel in load_split(ds, split):
                img = root / rel
                lbl = root / "labels" / img.parent.name / (img.stem + ".txt")
                if not lbl.is_file():
                    continue
                W, H = Image.open(img).size
                m = max(W, H)
                for line in lbl.read_text().splitlines():
                    p = line.split()
                    if len(p) != 5:
                        continue
                    bw, bh = float(p[3]) * W / m, float(p[4]) * H / m     # box size relative to the letterboxed side
                    by_cls[p[0]].append((min(bw, bh), max(bw, bh)))
        for c in sorted(by_cls, key=lambda x: int(x) if x.isdigit() else x):
            v = by_cls[c]
            s320 = [s * 320 / PATCH for s, _ in v]
            l320 = [l * 320 / PATCH for _, l in v]
            s1024 = [s * 1024 / PATCH for s, _ in v]
            lines.append(f"| {ds}{' (excluded)' if ds in rc.DROP else ''} | {names.get(c, c)} | {len(v)} | "
                         f"{statistics.median(s320):.2f} | {statistics.median(l320):.2f} | {statistics.median(s1024):.2f} | "
                         f"{sum(x < 1 for x in s320) / len(v):.0%} | {sum(x < 2 for x in s320) / len(v):.0%} |")
    (OUT / "table_tokens.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT / 'table_tokens.md'}")


# ── paper tables ─────────────────────────────────────────────────────────────────────────────
def _table(agg, datasets, columns, header_names, title):
    out = [f"### {title}", "", "| dataset (metric) | " + " | ".join(header_names) + " |",
           "|---|" + "---|" * len(columns)]
    for ds, task in datasets:
        row = agg.get(ds, {})
        best = max((row[c].mean for c in columns if c in row), default=None)
        cells = []
        for c in columns:
            s = row.get(c)
            cells.append("—" if s is None else (f"**{s.fmt()}**" if s.mean == best else s.fmt()))
        out.append(f"| {ds} ({rc.METRIC_LABEL[rc.HEADLINE[task]]}) | " + " | ".join(cells) + " |")
    return out + [""]


def _agg_by(runs, key):
    """{dataset: {key(run): Stat}} over seeds (headline metric)."""
    vals = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        v = rc.metric_of(r)
        if v is not None and r.seed in SEEDS:
            vals[r.dataset][key(r)][r.seed] = v
    out = {}
    for ds, per in vals.items():
        out[ds] = {}
        for k, by_seed in per.items():
            xs = list(by_seed.values())
            out[ds][k] = rc.Stat(statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0, len(xs), sorted(by_seed))
    return out


def cmd_tables(_):
    OUT.mkdir(parents=True, exist_ok=True)
    main_runs = [r for r in runs_in(runner.OUTDIR) if r.dataset not in rc.DROP]
    task = rc.task_of(main_runs)
    arm_sha = {a: sha256(arm_weights(a)) for a in ABLATION_ARMS if arm_weights(a)}
    sha_arm = {v: k for k, v in arm_sha.items()}
    md = ["# Paper tables (generated by experiments/paper.py tables)", "",
          f"Test-set headline metric, mean ± std over seeds {SEEDS}; bold = best in row; (n=k) = fewer than 3 seeds done. "
          f"Excluded datasets: {', '.join(rc.DROP)}. Locked split (contract amendment A1).", ""]

    # T1 main (C1): "ours" = MAIN_ARM checkpoint
    ours_sha = arm_sha.get(MAIN_ARM)
    def main_key(r):
        if r.backbone == "dinov3_dapt_b_fpn":
            return "ours" if r.weights_sha256 == ours_sha else None
        return r.backbone
    t1 = [r for r in main_runs if main_key(r)]
    cols = [b if b != "dinov3_dapt_b_fpn" else "ours" for b in MAIN_BACKBONES]
    names = [rc.display_name(b) if b != "ours" else f"DINOv3-DAPT ({MAIN_ARM})" for b in cols]
    for proto in ("frozen", "finetune"):
        agg = _agg_by([r for r in t1 if r.protocol == proto], main_key)
        dss = [(d, task[d]) for d in rc.ordered_datasets(agg)]
        md += _table(agg, dss, cols, names, f"T1 — main benchmark, {proto}")

    # T2 ablation (C2): arms, frozen
    t2 = [r for r in main_runs if r.backbone == "dinov3_dapt_b_fpn" and r.protocol == "frozen" and r.weights_sha256 in sha_arm]
    agg = _agg_by(t2, lambda r: sha_arm[r.weights_sha256])
    dss = [(d, task[d]) for d in rc.ordered_datasets(agg)]
    md += _table(agg, dss, ABLATION_ARMS, ABLATION_ARMS, f"T2 — DAPT sampling ablation (frozen, BUDGET={BUDGET})")
    if "uniform" in arm_sha:
        md += ["| Δ vs uniform (datasets better / mean Δ) | " + " | ".join(
            (lambda ds: f"{sum(d > 0 for d in ds)}/{len(ds)}, {statistics.mean(ds):+.3f}" if ds else "—")(
                [agg[d][a].mean - agg[d]["uniform"].mean for d, _ in dss if a in agg[d] and "uniform" in agg[d]])
            for a in ABLATION_ARMS) + " |", ""]

    # T3 resolution (C3): 320 px (E4 ablation) vs 1024 px (main table), frozen, detect datasets
    abl = runs_in(RES_ABL_DIR)
    def res_key(r):
        if r.backbone == "dinov3_dapt_b_fpn":
            return "ours" if r.weights_sha256 == ours_sha else None
        return r.backbone
    cols3 = [b if b != "dinov3_dapt_b_fpn" else "ours" for b in RES_ABL["backbones"]]
    lo = _agg_by([r for r in abl if r.protocol == "frozen" and r.dataset in RES_ABL["datasets"] and res_key(r) in cols3], res_key)
    hh = _agg_by([r for r in main_runs if r.protocol == "frozen" and r.dataset in RES_ABL["datasets"] and res_key(r) in cols3], res_key)
    md += ["### T3 — input resolution (frozen, detection mAP50): 320 px → 1024 px", "",
           "| dataset | " + " | ".join(rc.display_name(c) if c != "ours" else "ours" for c in cols3) + " |",
           "|---|" + "---|" * len(cols3)]
    for ds in RES_ABL["datasets"]:
        cells = []
        for c in cols3:
            a, b = lo.get(ds, {}).get(c), hh.get(ds, {}).get(c)
            cells.append(f"{a.mean:.3f} → {b.mean:.3f}" if a and b else (f"{a.mean:.3f} → —" if a else "—"))
        md.append(f"| {ds} | " + " | ".join(cells) + " |")
    md.append("")
    if (OUT / "table_tokens.md").is_file():
        md += ["(object-size-in-tokens table: `table_tokens.md`, from `paper.py tokens`)", ""]

    (OUT / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {OUT / 'tables.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    c = sub.add_parser("commands")
    c.add_argument("ids", nargs="*")
    sub.add_parser("tokens")
    sub.add_parser("tables")
    args = ap.parse_args()
    {"status": cmd_status, "commands": cmd_commands, "tokens": cmd_tokens, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
