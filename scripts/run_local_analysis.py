"""Recompute the offline thesis analysis in a new folder, preserving saved drafts.

Run with .venv/bin/python scripts/run_local_analysis.py from the project root.
Uses existing embeddings and reviewed citation data; no API or model downloads.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "outputs/local_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    destination.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv/matplotlib"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".venv/jupyter-runtime"))
    print(f"Local results: {destination}", flush=True)

    from batill.pipeline import validate_outputs
    from batill.thesis_analysis import run_analysis
    import numpy as np
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score

    validations = [validate_outputs(ROOT / "outputs" / name)
                   for name in ("corpus", "corpus_sections")]
    base = destination / "base_evidence"
    run_analysis(ROOT, base)
    comparisons = []
    with np.load(ROOT / "reports/thesis/evidence/text_partitions.npz") as original:
        with np.load(base / "text_partitions.npz") as fresh:
            for name in original.files:
                if name in fresh.files:
                    comparisons.append({"solution": name,
                        "ARI_against_saved": adjusted_rand_score(original[name], fresh[name]),
                        "identical_labels": bool(np.array_equal(original[name], fresh[name]))})
                else:
                    comparisons.append({"solution": name, "ARI_against_saved": None,
                                        "identical_labels": False})
    pd.DataFrame(comparisons).to_csv(destination / "comparison_with_saved.csv", index=False)

    # The revision's design explicitly reuses the frozen base memberships and
    # refits the additional sensitivity experiments. Keep that design unchanged.
    import thesis_v2_evidence as revision
    revised = destination / "thesis_v2"
    revision.REPORT = revised
    revision.OUT = revised / "evidence"
    revision.run()
    shutil.copy2(ROOT / "reports/thesis_v2/thesis_sections.md", revised)

    import verify_thesis_v2 as verification
    verification.OUT = revision.OUT
    verified = verification.verify()

    import build_thesis_v2 as word
    word.REPORT = revised
    word.EVIDENCE = revision.OUT
    word.word.EVIDENCE = revision.OUT
    tables = json.loads((revision.OUT / "tables.json").read_text())
    word.build(revised / "thesis_sections.md", revised / "thesis_sections.docx", tables)
    word.build(word.appendix_source(), revised / "supporting_appendix.docx", tables)

    versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (destination / "installed-packages.txt").write_text(versions)
    (destination / "run_summary.json").write_text(json.dumps({
        "python": sys.version, "executable": sys.executable,
        "embedding_validation": validations, "revision_verification": verified,
        "base_partition_comparisons": comparisons,
        "revision_base": "Original frozen memberships, as specified by the v2 workflow",
        "inference_run": False, "api_retrieval_run": False,
    }, indent=2) + "\n")
    print(f"Completed local analysis: {destination}", flush=True)


if __name__ == "__main__":
    main()
