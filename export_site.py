"""Turn saved analyses into data files for the results page in docs/.

Usage (from the workspace folder):
    python rootstep/export_site.py

Reads every runs/*.analysis.json, writes docs/data/<run_id>.json for each and
docs/data/index.json listing them. The page (docs/index.html) only reads
these files, so the published demo costs nothing to keep online.
"""
import json
from pathlib import Path

from rootstep.agent import first_thought
from rootstep.attribute import build_report, merge_scores
from rootstep.record import Run

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"
DATA_DIR = ROOT / "docs" / "data"
MAX_OUTPUT = 1500   # characters of command output kept per step on the page
LONG_REPLY = 1500   # replies longer than this are shown in full: they usually invent outputs
MAX_REPLY = 4000


def find_run(run_id: str) -> Path:
    for folder in (RUNS_DIR, RUNS_DIR / "reruns"):
        if (folder / f"{run_id}.json").exists():
            return folder / f"{run_id}.json"
    raise FileNotFoundError(run_id)


def thought_of(reply: str) -> str:
    """The thought behind the command that ran: the reply's first THOUGHT, minus any leaked </think> text."""
    return first_thought(reply).partition("</think>")[0].strip()


def page_data(run: Run, analysis: dict) -> dict:
    steps = []
    for s in run.steps:
        finished = s.action is None and s.observation_shown == ""
        output = s.observation_raw if s.action else s.observation_shown
        steps.append({
            "index": s.index,
            "thought": thought_of(s.raw_response),
            "action": s.action,
            "finish": finished,
            "exit_code": s.exit_code,
            "output": output[:MAX_OUTPUT] + ("\n…" if len(output) > MAX_OUTPUT else ""),
            "planted": s.planted,
            "reply_chars": len(s.raw_response),
            "reply": s.raw_response[:MAX_REPLY] + ("\n…" if len(s.raw_response) > MAX_REPLY else "")
                     if len(s.raw_response) > LONG_REPLY else None,
        })
    scores = [{k: v for k, v in sc.items() if k not in ("rerun_ids", "errors")} for sc in analysis["scores"]]
    return {
        "run": {"run_id": run.run_id, "task": run.task, "success": run.success, "finished": run.finished,
                "planted_step": run.planted_step,
                "model": run.config.get("model"), "max_steps": run.config.get("max_steps"),
                "cost_usd": round(run.total_cost_usd, 5), "steps": steps},
        "analysis": {"scores": scores,
                     "reruns": sum(sc["tries"] for sc in analysis["scores"]),
                     "point_of_no_return": analysis.get("point_of_no_return"),
                     "biggest_drop": analysis.get("biggest_drop"),
                     "cost_usd": analysis.get("cost_usd")},
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for old in DATA_DIR.glob("*.json"):   # the page shows exactly the analyses in runs/ now
        old.unlink()
    index = []
    for path in sorted(RUNS_DIR.glob("*.analysis.json")):
        analysis = json.loads(path.read_text())
        run = Run.load(find_run(analysis["run_id"]))
        if "point_of_no_return" not in analysis:   # made by an older analyze.py: recompute the findings
            analysis = build_report(run, merge_scores(analysis["scores"], []), analysis.get("cost_usd", 0.0))
        data = page_data(run, analysis)
        (DATA_DIR / f"{run.run_id}.json").write_text(json.dumps(data, indent=1))
        index.append({"run_id": run.run_id, "task": run.task, "success": run.success,
                      "planted_step": run.planted_step,
                      "steps": len(run.steps), "reruns": data["analysis"]["reruns"],
                      "point_of_no_return": analysis["point_of_no_return"]})
        print(f"exported {run.run_id} ({len(run.steps)} steps, {data['analysis']['reruns']} re-runs)")
    (DATA_DIR / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {DATA_DIR / 'index.json'} ({len(index)} runs)")


if __name__ == "__main__":
    main()
