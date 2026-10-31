#!/usr/bin/env python3
"""Run the golden-set evaluation against the configured local RAG stack.

Examples:
  python -m eval.run_eval --dataset eval/datasets/golden.template.jsonl --mode retrieval
  python -m eval.run_eval --dataset eval/datasets/my_golden.jsonl --mode full --ragas

The template is illustrative. Replace its reference contexts and relevant chunk IDs
with expert-reviewed labels from your own indexed corpus before treating scores as
meaningful.
"""
import argparse, asyncio, csv, json, os, sys, time, math
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eval.schema import load_golden
from eval.metrics import hit_rate_at_k, reciprocal_rank, ndcg_at_k, mean
from rag.query_understanding import classify_query, assess_corpus_compatibility
from rag.lexical import domain_counts


def route_for(row, min_confidence):
    classification=classify_query(row["question"])
    compatibility=assess_corpus_compatibility(classification,domain_counts(row["tenant_id"]),min_confidence)
    route={"proceed":"answer","domain_mismatch":"domain_mismatch","clarify":"clarify"}[compatibility["status"]]
    return classification,compatibility,route


async def run(rows, mode, min_confidence):
    from rag.answer import generate_and_verify
    from rag.retrieve import search
    output=[]
    for row in rows:
        case_started=time.perf_counter()
        classification,compatibility,predicted_route=route_for(row,min_confidence)
        retrieved=[]; answer=""; verified=False; sources=[]
        effective_filters=dict(row.get("filters") or {})
        if not effective_filters.get("domains") and classification["domain"] in compatibility.get("available_domains", {}):
            effective_filters["domains"]=[classification["domain"]]
        if predicted_route == "answer":
            if mode == "full":
                result=await generate_and_verify(row["question"],tenant_id=row["tenant_id"],
                    filters=effective_filters,include_contexts=True)
                answer=result.get("answer","")
                verified=bool(result.get("verified")) and result.get("verification")!="no_evidence"
                sources=result.get("sources",[])
                retrieved=result.get("retrieved_contexts",[])
                retrieved_ids=result.get("retrieved_chunk_ids",[])
            else:
                chunks=search(row["question"],tenant_id=row["tenant_id"],filters=effective_filters)
                retrieved=[c.get("text","") for c in chunks]
                retrieved_ids=[c.get("chunk_id","") for c in chunks]
        else:
            retrieved_ids=[]
        relevant=row.get("relevant_chunk_ids") or []
        rec={"id":row["id"],"question":row["question"],"category":row.get("category","uncategorized"),"expected_domain":row["expected_domain"],
             "predicted_domain":classification["domain"],"domain_correct":classification["domain"]==row["expected_domain"],
             "expected_route":row["expected_route"],"predicted_route":predicted_route,
             "route_correct":predicted_route==row["expected_route"],"domain_confidence":classification["domain_confidence"],
             "retrieved_chunk_ids":retrieved_ids,"retrieved_contexts":retrieved,"answer":answer,
             "ground_truth":row.get("ground_truth",""),"reference_contexts":row.get("reference_contexts",[]),
             "relevant_chunk_ids":relevant,"hit_at_5":hit_rate_at_k(retrieved_ids,relevant,5),
             "mrr":reciprocal_rank(retrieved_ids,relevant),"ndcg_at_5":ndcg_at_k(retrieved_ids,relevant,5),
             "citation_integrity_passed":verified if mode=="full" and predicted_route=="answer" and retrieved else None,
             "latency_ms":round((time.perf_counter()-case_started)*1000,1),
             "sources":sources,"domain_check":compatibility}
        output.append(rec)
    return output


def run_ragas(records, out_dir):
    """Optional RAGAS 0.2.x evaluation; uses the project's OpenAI-compatible endpoint."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from rag import config
    except ImportError as exc:
        raise RuntimeError("Install optional evaluation dependencies: pip install -r requirements-eval.txt") from exc
    usable=[r for r in records if r["predicted_route"]=="answer" and r["answer"] and r["retrieved_contexts"] and r["ground_truth"].strip()]
    if not usable: raise RuntimeError("No answer-route rows have answers, retrieved contexts and ground_truth for RAGAS")
    ds=Dataset.from_list([{"question":r["question"],"answer":r["answer"],
        "contexts":r["retrieved_contexts"],"ground_truth":r["ground_truth"]} for r in usable])
    api_key=os.getenv("RAGAS_API_KEY") or config.LLM_API_KEY
    base_url=os.getenv("RAGAS_BASE_URL") or (config.LLM_BASE_URL if config.LLM_BASE_URL else None)
    model=os.getenv("RAGAS_LLM_MODEL") or config.LLM_MODEL
    if not api_key or not model: raise RuntimeError("Set RAGAS_API_KEY/RAGAS_LLM_MODEL or configure LLM_API_KEY/LLM_MODEL")
    llm=ChatOpenAI(model=model,api_key=api_key,base_url=base_url,temperature=0)
    embeddings=OpenAIEmbeddings(model=os.getenv("RAGAS_EMBEDDING_MODEL","text-embedding-3-small"),
        api_key=api_key,base_url=os.getenv("RAGAS_EMBEDDING_BASE_URL") or base_url)
    result=evaluate(ds,metrics=[faithfulness,answer_relevancy,context_precision,context_recall,answer_correctness],
                    llm=llm,embeddings=embeddings)
    frame=result.to_pandas()
    frame.to_csv(out_dir/"ragas_per_question.csv",index=False)
    summary={k:float(v) for k,v in result.items() if isinstance(v,(int,float))}
    return {"scored_questions":len(usable),"metrics":summary}


def main():
    ap=argparse.ArgumentParser(description="Golden dataset and RAGAS evaluation")
    ap.add_argument("--dataset",default="eval/datasets/golden.template.jsonl")
    ap.add_argument("--mode",choices=["retrieval","full"],default="retrieval")
    ap.add_argument("--ragas",action="store_true",help="Run RAGAS metrics (requires --mode full)")
    ap.add_argument("--min-domain-confidence",type=float,default=float(os.getenv("QUERY_DOMAIN_MIN_CONFIDENCE","0.62")))
    ap.add_argument("--output-dir",default="eval/results")
    args=ap.parse_args()
    if args.ragas and args.mode!="full": ap.error("--ragas requires --mode full")
    rows=load_golden(args.dataset)
    out_dir=Path(args.output_dir);out_dir.mkdir(parents=True,exist_ok=True)
    records=asyncio.run(run(rows,args.mode,args.min_domain_confidence))
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path=out_dir/f"evaluation_{stamp}.json"
    csv_path=out_dir/f"evaluation_{stamp}.csv"
    json_path.write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding="utf-8")
    fields=["id","predicted_domain","expected_domain","domain_correct","predicted_route","expected_route","route_correct","hit_at_5","mrr","ndcg_at_5","citation_integrity_passed"]
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(records)
    def aggregate(group):
        return {"cases":len(group),
            "domain_accuracy":mean([float(r["domain_correct"]) for r in group]),
            "route_accuracy":mean([float(r["route_correct"]) for r in group]),
            "hit_rate_at_5":mean([r["hit_at_5"] for r in group]),
            "mrr":mean([r["mrr"] for r in group]),
            "ndcg_at_5":mean([r["ndcg_at_5"] for r in group]),
            "mean_latency_ms":mean([r["latency_ms"] for r in group]),
            "citation_integrity_rate":mean([float(r["citation_integrity_passed"]) for r in group if r["citation_integrity_passed"] is not None])}
    domains=sorted({r["expected_domain"] for r in records})
    categories=sorted({r.get("category","uncategorized") for r in records})
    latencies=sorted(r["latency_ms"] for r in records)
    p95=latencies[max(0,math.ceil(0.95*len(latencies))-1)] if latencies else None
    summary={"dataset":args.dataset,"mode":args.mode,"cases":len(records),**aggregate(records),
        "p95_latency_ms":p95,
        "by_expected_domain":{d:aggregate([r for r in records if r["expected_domain"]==d]) for d in domains},
        "by_category":{c:aggregate([r for r in records if r.get("category","uncategorized")==c]) for c in categories},
        "artifacts":{"json":str(json_path),"csv":str(csv_path)}}
    if args.ragas: summary["ragas"]=run_ragas(records,out_dir)
    summary_path=out_dir/f"summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
