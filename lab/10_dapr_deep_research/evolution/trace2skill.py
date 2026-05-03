from __future__ import annotations

import json
from pathlib import Path


class SkillConsolidator:
    def __init__(self, persist_dir: str | Path):
        self.dir = Path(persist_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def consolidate(self, trajectories: list[dict]) -> dict:
        error_patterns = []
        success_patterns = []
        demos = []
        for traj in trajectories:
            steps = traj if isinstance(traj, list) else traj.get("trajectory", [])
            for step in steps:
                reasoning = step.get("reasoning", "")
                output = step.get("output", "")
                code = step.get("code", "")
                if "error" in str(output).lower() or "fail" in str(output).lower():
                    error_patterns.append({"symptom": output[:200], "reasoning": reasoning[:300]})
                elif reasoning and len(reasoning) > 30:
                    success_patterns.append({"reasoning": reasoning[:300], "code": code[:200] if code else ""})
            final = steps[-1] if steps else {}
            if final.get("output"):
                demos.append({"reasoning": final.get("reasoning", "")[:500], "output": str(final.get("output", ""))[:500]})
        return {"error_patterns": error_patterns[:10], "success_patterns": success_patterns[:10], "demonstrations": demos[:5], "n_trajectories": len(trajectories)}

    def save_skill(self, name: str, skill: dict):
        (self.dir / f"{name}.json").write_text(json.dumps(skill, indent=2, default=str))

    def load_skills(self) -> list[dict]:
        skills = []
        for f in sorted(self.dir.glob("*.json"), reverse=True):
            skills.append(json.loads(f.read_text()))
        return skills
