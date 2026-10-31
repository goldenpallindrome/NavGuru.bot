"""Fast, deterministic query intent/domain analysis and corpus compatibility checks.

This is a routing signal, not a medical/legal/financial safety validator. Unknown or
low-confidence queries are not rejected; retrieval evidence remains authoritative.
"""
from collections import Counter
import re

DOMAIN_TERMS = {
    "medical": {"diabetes","symptom","symptoms","disease","diagnosis","diagnose","medicine","medication","drug","treatment","treat","doctor","patient","blood pressure","heart rate","cancer","infection","clinical","medical","health","therapy","dosage","side effect","pregnancy","vaccine","fever","pain","insulin","glucose"},
    "finance": {"revenue","profit","loss","balance sheet","cash flow","invoice","payment","financial","finance","accounting","tax","taxation","budget","expense","investment","interest rate","loan","credit","debit","audit","equity","asset","liability","dividend","portfolio","banking"},
    "legal": {"contract","agreement","clause","litigation","lawsuit","legal","law","regulation","compliance","liability","jurisdiction","statute","court","rights","patent","trademark"},
    "hr": {"employee","employment","payroll","leave policy","annual leave","hiring","recruitment","performance review","termination","benefits","human resources","hr policy","workplace"},
    "technology": {"api","software","database","server","network","code","programming","architecture","deployment","authentication","security vulnerability","cloud","machine learning","model","debugging"},
    "insurance": {"insurance","premium","claim","deductible","coverage","policyholder","underwriting","beneficiary"},
    "education": {"curriculum","student","course","exam","syllabus","academic","university","school","learning outcome","tuition"},
}
INTENTS = {
    "summarization": ("summarize","summary","overview","recap","key points"),
    "comparison": ("compare","comparison","difference","versus"," vs "),
    "calculation": ("calculate","compute","how much","total","percentage","rate of"),
    "procedural": ("how do i","how to","steps","process","procedure","configure","install"),
    "explanation": ("explain","what does","meaning of","why does","describe"),
    "fact_lookup": ("what is","when did","who is","which","where is","list","show me"),
}

def classify_query(question: str) -> dict:
    q = re.sub(r"\s+", " ", question.lower()).strip()
    scores = Counter()
    matches = {}
    for domain, terms in DOMAIN_TERMS.items():
        found = [term for term in terms if term in q]
        if found:
            scores[domain] = sum(2 if " " in term else 1 for term in found)
            matches[domain] = found[:8]
    ordered = scores.most_common()
    domain = ordered[0][0] if ordered else "unknown"
    total = sum(scores.values())
    confidence = (scores[domain] / total) if total else 0.0
    # Confidence reflects keyword signal strength, but never treats keyword rules as proof.
    if ordered and scores[domain] >= 2:
        confidence = min(0.99, 0.55 + 0.12 * scores[domain] + 0.08 * (scores[domain] / max(1,total)))
    elif ordered:
        confidence = 0.68
    else:
        confidence = 0.0
    intent = "information_seeking"
    for name, phrases in INTENTS.items():
        if any(p in q for p in phrases):
            intent = name
            break
    entities = []
    for vals in matches.values():
        entities.extend(vals)
    return {"domain": domain, "domain_confidence": round(confidence, 3),
            "intent": intent, "matched_terms": entities[:10],
            "requires_clarification": len(ordered) > 1 and ordered[0][1] == ordered[1][1]}

def assess_corpus_compatibility(classification: dict, domain_counts: dict, min_confidence: float = 0.62) -> dict:
    requested = classification.get("domain", "unknown")
    available = {k: int(v) for k,v in domain_counts.items() if int(v) > 0 and k != "general"}
    if requested == "unknown" or classification.get("domain_confidence", 0) < min_confidence or not available:
        return {"status": "proceed", "reason": "domain_unclassified_or_corpus_unlabelled", "available_domains": available}
    if requested in available:
        return {"status": "proceed", "reason": "matching_domain_available", "available_domains": available}
    # A query can be cross-domain; do not block if it explicitly signals multiple domains.
    if classification.get("requires_clarification"):
        return {"status": "clarify", "reason": "ambiguous_multi_domain_query", "available_domains": available}
    return {"status": "domain_mismatch", "reason": "requested_domain_not_in_corpus", "available_domains": available}
