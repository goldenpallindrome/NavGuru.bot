"""Golden dataset loading and validation."""
import json
from pathlib import Path

REQUIRED={"id","question","tenant_id","expected_domain","expected_route"}
ROUTES={"answer","domain_mismatch","clarify"}


def load_golden(path):
    rows=[]
    for line_no,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip() or line.lstrip().startswith("#"): continue
        try: row=json.loads(line)
        except json.JSONDecodeError as exc: raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}")
        missing=REQUIRED-set(row)
        if missing: raise ValueError(f"{path}:{line_no}: missing fields {sorted(missing)}")
        if row["expected_route"] not in ROUTES: raise ValueError(f"{path}:{line_no}: invalid expected_route")
        if not isinstance(row["question"],str) or not row["question"].strip(): raise ValueError(f"{path}:{line_no}: empty question")
        row.setdefault("filters",{})
        row.setdefault("ground_truth"," ")
        row.setdefault("reference_contexts",[])
        row.setdefault("relevant_chunk_ids",[])
        rows.append(row)
    if not rows: raise ValueError("Golden dataset is empty")
    return rows
