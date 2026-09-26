#!/usr/bin/env python3
"""smoke_windows.py — smoke-test the Windows-side scripts (native Windows Python), after smoke_wsl.sh.

  C:\\ProgramData\\miniconda3\\python.exe experiments\\smoke_windows.py [groups...]
  groups: data (12/13 on smoke_out/frames; needs smoke_wsl 'data'), splits (make_splits --check, read-only),
          reports (5 report scripts on the smoke results; needs smoke_wsl 'bench'), paper (paper.py on smoke results)
  default: all groups. PASS/FAIL per step -> smoke_out/SUMMARY_windows.txt, logs -> smoke_out/logs/win_*.log

Nothing here writes to real artifacts: frames/sidecars go to smoke_out/, figures/HTML to smoke_out/, paper.py
reads the smoke results via PAPER_RESULTS_DIR / PAPER_RES_ABL_DIR and writes experiments/out_smoke/.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

DT = Path(__file__).resolve().parents[1]
BENCH = Path(os.environ.get("BENCH_ROOT", r"D:\Conceivable-SharedData01-23Jun2026"))
S = DT / "smoke_out"
PY = sys.executable
SMOKE_RES = BENCH / "stage1_out" / "benchmark_results_smoke"
SMOKE_ABL = BENCH / "stage1_out" / "benchmark_results_imgsz320_smoke"
SUM = S / "SUMMARY_windows.txt"


def run(name, cmd, cwd=DT, env=None, check=None):
    """run a command; PASS if rc == 0 and (optional) check(log_text) is truthy."""
    (S / "logs").mkdir(parents=True, exist_ok=True)
    log = S / "logs" / f"win_{name}.log"
    t0 = time.time()
    e = dict(os.environ, PYTHONIOENCODING="utf-8", **(env or {}))
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=cwd, env=e, stdout=f, stderr=subprocess.STDOUT).returncode
    ok = rc == 0 and (check is None or check(log.read_text(encoding="utf-8", errors="replace")))
    line = f"{name:44s} {'PASS' if ok else f'FAIL(rc={rc})':10s} {time.time() - t0:5.0f}s  logs/{log.name}"
    print(line, flush=True)
    with open(SUM, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return ok


def main():
    groups = sys.argv[1:] or ["data", "splits", "reports", "paper"]
    with open(SUM, "a", encoding="utf-8") as f:          # append: groups may run in separate invocations
        f.write(f"smoke_windows started {time.strftime('%F %T')}  groups: {groups}\n")

    if "data" in groups:
        run("data_12_motion_energy", [PY, "-u", "scripts/12_motion_energy.py", "--manifest", "manifests/train_videos.txt",
                                      "--split", "train", "--limit", "1", "--workers", "1",
                                      "--frames-root", str(S / "frames")],
            check=lambda t: "err=0" in t)
        run("data_13_frame_weights", [PY, "-u", "scripts/13_frame_weights.py", "--frames-root", str(S / "frames"),
                                      "--split", "train"],
            check=lambda t: "frame_weights" in t or (S / "frames" / "train" / "frame_weights.npy").is_file())

    if "splits" in groups:
        for prof in ("recording", "acquisition"):
            run(f"make_splits_check_{prof}", [PY, "stage1_out/benchmark/dataset/make_splits.py", "--check", "--profile", prof],
                cwd=BENCH, check=lambda t: '"near_dup_new"' in t)

    if "reports" in groups:
        plots = S / "plots"
        for prof, res in (("recording", SMOKE_RES), ("acquisition", SMOKE_RES)):
            run(f"report_summarize_{prof}", [PY, "summarize_bench.py", str(res), "--split-profile", prof], cwd=BENCH,
                check=lambda t: "FROZEN" in t)
        # the smoke results hold TWO DAPT checkpoints (default + the E3 override), so single-"ours" reports must be
        # told which one; without --ours they must refuse (guard against silently picking one)
        ours = ["--ours", "dinov3_dapt_b_fpn@dapt"]
        run("report_where_ours_refuses_ambiguous", [PY, "-c", "import subprocess,sys; r=subprocess.run([sys.executable,'where_ours.py',r'%s'],capture_output=True,text=True); print(r.stdout+r.stderr); sys.exit(0 if r.returncode!=0 and 'several DAPT variants' in (r.stdout+r.stderr) else 1)" % SMOKE_RES],
            cwd=BENCH)
        run("report_where_ours", [PY, "where_ours.py", str(SMOKE_RES)] + ours, cwd=BENCH, check=lambda t: "ours =" in t)
        run("report_plot_bench", [PY, "plot_bench.py", str(SMOKE_RES), "--plots", str(plots)], cwd=BENCH,
            check=lambda t: "bench_bars.png" in t)
        run("report_make_ppt_figs", [PY, "make_ppt_figs.py", str(SMOKE_RES), "--plots", str(plots)] + ours, cwd=BENCH,
            check=lambda t: "ppt_results_heatmap.png" in t)
        run("report_make_report", [PY, "make_report.py", str(SMOKE_RES), "--plots", str(plots),
                                   "--out", str(S / "report_smoke.html")] + ours, cwd=BENCH, check=lambda t: "wrote" in t)
        run("report_ablation_dir_summarize", [PY, "summarize_bench.py", str(SMOKE_ABL)], cwd=BENCH,
            check=lambda t: "FROZEN" in t)

    if "paper" in groups:
        env = {"PAPER_RESULTS_DIR": str(SMOKE_RES), "PAPER_RES_ABL_DIR": str(SMOKE_ABL)}
        run("paper_status", [PY, "experiments/paper.py", "status"], env=env, check=lambda t: "E0" in t and "E4" in t)
        run("paper_commands", [PY, "experiments/paper.py", "commands"], env=env, check=lambda t: "### E0" in t)
        run("paper_tokens", [PY, "experiments/paper.py", "tokens"], check=lambda t: "needle_tip" in t)
        run("paper_tables", [PY, "experiments/paper.py", "tables"], env=env, check=lambda t: "T0" in t and "T3" in t)

    with open(SUM, "a", encoding="utf-8") as f:
        f.write(f"smoke_windows finished {time.strftime('%F %T')}\n")


if __name__ == "__main__":
    main()
