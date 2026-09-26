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
from benchmark.weights_id import weights_identity  # noqa: E402

OUT = DT / "experiments" / "out"

# ── the plan ─────────────────────────────────────────────────────────────────────────────────
# E0 (auxiliary, answers "does a ViT foundation model beat a CNN here at all?" BEFORE spending on E1):
# CNN vs raw DINOv3 vs the already-shipped DAPT checkpoint (uniform crops, 20k it; = DINOV3_DAPT_B_WEIGHTS
# default, sha d7282330), evaluated on the "acquisition" split profile -- test images come from days /
# patient cases / sessions never seen in training, so the train/test gap is deliberately larger.
E0 = dict(split_profile="acquisition",
          backbones=["resnet50_fpn", "dinov3_b_fpn", "dinov3_dapt_b_fpn"],
          datasets=["cellasp", "holding_pip", "routine2_coc", "cvit_incubator", "cvit_workstation"])
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
RES_ABL_DIR = Path(os.environ.get("PAPER_RES_ABL_DIR", BENCH / "stage1_out" / f"benchmark_results_imgsz{RES_ABL['imgsz']}"))
RESULTS_DIR = Path(os.environ.get("PAPER_RESULTS_DIR", runner.OUTDIR))   # overridable for smoke tests
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


def runs_in(results_dir: Path, split_profile: str = "recording"):
    return (rc.load_runs(results_dir, include_drop=True, quiet=True, split_profile=split_profile)
            if results_dir.is_dir() else [])


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
    main_runs = runs_in(RESULTS_DIR)
    abl_runs = runs_in(RES_ABL_DIR)
    rows = []

    # E0 — auxiliary: is a ViT foundation model worth it vs a CNN? (held-out acquisition batches)
    e0_runs = runs_in(RESULTS_DIR, E0["split_profile"])
    shipped_path, shipped_sha = weights_identity("dinov3_dapt_b_fpn")
    per_bb = len(E0["datasets"]) * 2 * len(SEEDS)
    done, cmds = 0, []
    base = (f"python run_benchmark_server.py --all-seeds --split-profile {E0['split_profile']} "
            f"--dataset {' '.join(E0['datasets'])} --skip-done")
    for bb in E0["backbones"]:
        n = count([r for r in e0_runs if r.dataset in E0["datasets"]], bb,
                  shipped_sha if bb == "dinov3_dapt_b_fpn" else None)
        done += n
        if n < per_bb:
            env = f"DINOV3_DAPT_B_WEIGHTS={to_posix(Path(shipped_path))} " if bb == "dinov3_dapt_b_fpn" else ""
            cmds.append(f"{env}{base} --backbone {bb}")
    rows.append(("E0", f"auxiliary ViT-vs-CNN check on held-out days/cases: {len(E0['backbones'])} backbones x "
                       f"{len(E0['datasets'])} datasets x 2 protocols (shipped DAPT weights {shipped_sha[:8]})",
                 done, per_bb * len(E0["backbones"]),
                 [f"cd {to_posix(BENCH)}", "# ~30 runs; measure one 1024-px detection run first"] + cmds if cmds else [],
                 "no E1 needed; result decides whether E1-E4 are worth running (PAPER.md)"))

    # (single training seed + test-set bootstrap CIs -- contract amendment A3; SEEDS = runner.SEEDS = [42])
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
    rows.append(("E2", "main benchmark: 6 backbones x 9 datasets x 2 protocols x seed 42 (detect 1024 px)", done, total,
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
    rows.append(("E3", f"ablation: DAPT arms x 9 datasets x frozen x seed 42", done, per_arm * len(ABLATION_ARMS),
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
    rows.append(("E4", f"resolution ablation: 4 backbones x {len(RES_ABL['datasets'])} detect datasets x frozen x seed 42 @ {RES_ABL['imgsz']} px",
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
def _table(agg, proto, datasets, columns, header_names, title, extra=None):
    """agg: {(proto, ds): {col: Stat}}; extra: optional (header, fn(ds) -> str) appended column."""
    hdr = ["dataset (metric)"] + header_names + ([extra[0]] if extra else [])
    out = [f"### {title}", "", "| " + " | ".join(hdr) + " |", "|---" * len(hdr) + "|"]
    for ds, task in datasets:
        row = agg.get((proto, ds), {})
        best = max((row[c].mean for c in columns if c in row), default=None)
        cells = ["—" if c not in row else (f"**{row[c].fmt()}**" if row[c].mean == best else row[c].fmt())
                 for c in columns]
        if extra:
            cells.append(extra[1](ds))
        out.append(f"| {ds} ({rc.METRIC_LABEL[rc.HEADLINE[task]]}) | " + " | ".join(cells) + " |")
    return out + [""]


def _delta_summary(deltas):
    ds = [d for d in deltas if d]
    if not ds:
        return "—"
    above = sum(d[1] is not None and d[1] > 0 for d in ds)
    return f"better on {sum(d[0] > 0 for d in ds)}/{len(ds)}, CI>0 on {above}/{len(ds)}, mean Δ {sum(d[0] for d in ds) / len(ds):+.3f}"


def cmd_tables(_):
    OUT.mkdir(parents=True, exist_ok=True)
    main_runs = [r for r in runs_in(RESULTS_DIR) if r.dataset not in rc.DROP and r.seed in SEEDS]
    task = rc.task_of(main_runs)
    arm_sha = {a: sha256(arm_weights(a)) for a in ABLATION_ARMS if arm_weights(a)}
    sha_arm = {v: k for k, v in arm_sha.items()}
    ours_sha = arm_sha.get(MAIN_ARM)
    md = ["# Paper tables (generated by experiments/paper.py tables)", "",
          f"Test-set headline metric, seed {SEEDS}. Single seed: value [test-set bootstrap 95% CI, 1000 resamples]; "
          "Δ columns: paired bootstrap on the same test images, * = CI excludes 0 (contract amendment A3). "
          f"Bold = best in row. Excluded datasets: {', '.join(rc.DROP)}. Locked split (A1); detection at 1024 px (A2).", ""]

    def bb_key(r):                                        # DAPT runs count only for the MAIN_ARM checkpoint
        if r.backbone == "dinov3_dapt_b_fpn":
            return "ours" if r.weights_sha256 == ours_sha else None
        return r.backbone

    # T0 auxiliary (E0): CNN vs ViT on held-out acquisition batches
    _, shipped_sha = weights_identity("dinov3_dapt_b_fpn")
    e0 = [r for r in runs_in(RESULTS_DIR, E0["split_profile"]) if r.dataset in E0["datasets"] and r.seed in SEEDS]
    e0_key = lambda r: ("dapt" if r.weights_sha256 == shipped_sha else None) if r.backbone == "dinov3_dapt_b_fpn" else r.backbone
    agg0, idx0 = rc.aggregate(e0, key=e0_key), rc.index_runs(e0, key=e0_key)
    task0 = rc.task_of(e0)
    cols0 = ["resnet50_fpn", "dinov3_b_fpn", "dapt"]
    names0 = ["CNN (ResNet-50)", "DINOv3 (raw)", f"DINOv3-DAPT (shipped, {shipped_sha[:8]})"]
    for proto in ("frozen", "finetune"):
        dss = [(d, task0[d]) for d in rc.ordered_datasets(d for p, d in agg0 if p == proto)]
        c_raw = {d: rc.compare(idx0, proto, d, "dinov3_b_fpn", "resnet50_fpn") for d, _ in dss}
        c_dapt = {d: rc.compare(idx0, proto, d, "dapt", "resnet50_fpn") for d, _ in dss}
        md += _table(agg0, proto, dss, cols0, names0 + ["raw − CNN"],
                     f"T0 — auxiliary: ViT vs CNN on held-out days/cases (split profile `{E0['split_profile']}`), {proto}",
                     extra=("DAPT − CNN", lambda d: f"{rc.fmt_delta(c_raw[d])} | {rc.fmt_delta(c_dapt[d])}"))
        md += [f"DAPT vs CNN ({proto}): {_delta_summary(c_dapt.values())}; raw vs CNN: {_delta_summary(c_raw.values())}", ""]

    # T1 main (C1)
    cols = [b if b != "dinov3_dapt_b_fpn" else "ours" for b in MAIN_BACKBONES]
    names = [rc.display_name(b) if b != "ours" else f"DINOv3-DAPT ({MAIN_ARM})" for b in cols]
    agg, idx = rc.aggregate(main_runs, key=bb_key), rc.index_runs(main_runs, key=bb_key)
    for proto in ("frozen", "finetune"):
        dss = [(d, task[d]) for d in rc.ordered_datasets(d for p, d in agg if p == proto)]
        cmp = {d: rc.compare(idx, proto, d, "ours", "dinov3_b_fpn") for d, _ in dss}
        md += _table(agg, proto, dss, cols, names, f"T1 — main benchmark, {proto}",
                     extra=("ours − DINOv3-raw", lambda d: rc.fmt_delta(cmp[d])))
        md += [f"ours vs DINOv3-raw ({proto}): {_delta_summary(cmp.values())}", ""]

    # T2 ablation (C2): arms, frozen, paired Δ vs uniform
    t2 = [r for r in main_runs if r.backbone == "dinov3_dapt_b_fpn" and r.protocol == "frozen" and r.weights_sha256 in sha_arm]
    arm_key = lambda r: sha_arm.get(r.weights_sha256)
    agg2, idx2 = rc.aggregate(t2, key=arm_key), rc.index_runs(t2, key=arm_key)
    dss = [(d, task[d]) for d in rc.ordered_datasets(d for p, d in agg2)]
    md += _table(agg2, "frozen", dss, ABLATION_ARMS, ABLATION_ARMS, f"T2 — DAPT sampling ablation (frozen, BUDGET={BUDGET})")
    if "uniform" in arm_sha:
        others = [a for a in ABLATION_ARMS if a != "uniform"]
        md += ["| dataset | " + " | ".join(f"{a} − uniform" for a in others) + " |", "|---" * (len(others) + 1) + "|"]
        summ = {a: [] for a in others}
        for d, _ in dss:
            cells = []
            for a in others:
                c = rc.compare(idx2, "frozen", d, a, "uniform")
                summ[a].append(c)
                cells.append(rc.fmt_delta(c))
            md.append(f"| {d} | " + " | ".join(cells) + " |")
        md += ["| **summary** | " + " | ".join(_delta_summary(summ[a]) for a in others) + " |", ""]

    # T3 resolution (C3): 320 px (E4 ablation) vs 1024 px (main table), frozen, detect datasets
    cols3 = [b if b != "dinov3_dapt_b_fpn" else "ours" for b in RES_ABL["backbones"]]
    sel = lambda runs: [r for r in runs if r.protocol == "frozen" and r.dataset in RES_ABL["datasets"] and bb_key(r) in cols3]
    lo = rc.aggregate(sel(runs_in(RES_ABL_DIR)), key=bb_key)
    hh = rc.aggregate(sel(main_runs), key=bb_key)
    md += ["### T3 — detection input resolution (frozen, mAP50): 320 px → 1024 px", "",
           "| dataset | " + " | ".join(rc.display_name(c) if c != "ours" else "ours" for c in cols3) + " |",
           "|---" * (len(cols3) + 1) + "|"]
    for ds in RES_ABL["datasets"]:
        cells = []
        for c in cols3:
            a, b = lo.get(("frozen", ds), {}).get(c), hh.get(("frozen", ds), {}).get(c)
            cells.append(f"{a.mean:.3f} → {b.mean:.3f} ({b.mean - a.mean:+.3f})" if a and b
                         else (f"{a.mean:.3f} → —" if a else (f"— → {b.mean:.3f}" if b else "—")))
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
