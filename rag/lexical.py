import sqlite3
from pathlib import Path
from . import config

def db():
    Path("/app/data").mkdir(parents=True, exist_ok=True)
    con=sqlite3.connect("/app/data/lexical.sqlite3",timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    expected={"chunk_id","document_id","filename","page","tenant_id","document_type","tags","extraction","chunk_type","domain","text"}
    existing={r[1] for r in con.execute("PRAGMA table_info(chunk_fts)").fetchall()}
    if existing and existing != expected:
        # FTS schema migration: rebuild the index after deployment/re-ingestion.
        con.execute("DROP TABLE chunk_fts")
    con.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
      chunk_id UNINDEXED, document_id UNINDEXED, filename UNINDEXED, page UNINDEXED,
      tenant_id UNINDEXED, document_type UNINDEXED, tags UNINDEXED,
      extraction UNINDEXED, chunk_type UNINDEXED, domain UNINDEXED, text, tokenize='porter unicode61'
    )""")
    return con

def replace_document(document_id, rows):
    with db() as con:
        con.execute("DELETE FROM chunk_fts WHERE document_id=?",(document_id,))
        con.executemany(
          "INSERT INTO chunk_fts(chunk_id,document_id,filename,page,tenant_id,document_type,tags,extraction,chunk_type,domain,text) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
          [(r["chunk_id"],r["document_id"],r["filename"],str(r["page"]),r.get("tenant_id","default"),
            r.get("document_type","general")," ".join(r.get("tags",[])),r.get("extraction","native"),
            r.get("chunk_type","text_or_table"),r.get("domain","general"),r["text"]) for r in rows]
        )

def search_bm25(question,limit=50,tenant_id="default",filters=None):
    filters=filters or {}
    terms=[x.replace('"','""') for x in question.split() if len(x)>2][:15]
    if not terms:return []
    match=" OR ".join(f'"{x}"' for x in terms)
    clauses=["tenant_id=?"]; args=[tenant_id]
    def in_filter(column,key):
        vals=filters.get(key) or []
        if vals:
            clauses.append(column+" IN ("+",".join("?" for _ in vals)+")"); args.extend(vals)
    in_filter("document_id","document_ids")
    in_filter("filename","filenames")
    in_filter("extraction","extraction_methods")
    in_filter("chunk_type","chunk_types")
    in_filter("document_type","document_types")
    in_filter("domain","domains")
    if filters.get("pages"):
        vals=[str(v) for v in filters["pages"]]
        clauses.append("page IN ("+",".join("?" for _ in vals)+")"); args.extend(vals)
    if filters.get("page_from") is not None: clauses.append("CAST(page AS INTEGER)>=?");args.append(filters["page_from"])
    if filters.get("page_to") is not None: clauses.append("CAST(page AS INTEGER)<=?");args.append(filters["page_to"])
    for tag in filters.get("tags",[]): clauses.append("(' ' || tags || ' ') LIKE ?");args.append("% "+tag+" %")
    try:
        with db() as con:
            rows=con.execute(
              "SELECT chunk_id,document_id,filename,page,tenant_id,document_type,tags,extraction,chunk_type,domain,text,bm25(chunk_fts) "
              "FROM chunk_fts WHERE chunk_fts MATCH ? AND "+" AND ".join(clauses)+
              " ORDER BY bm25(chunk_fts) LIMIT ?",[match,*args,limit]
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [{"chunk_id":r[0],"document_id":r[1],"filename":r[2],"page":int(r[3]),
             "tenant_id":r[4],"document_type":r[5],"tags":r[6].split(),"extraction":r[7],
             "chunk_type":r[8],"domain":r[9],"text":r[10],"bm25_score":float(r[11])} for r in rows]


def domain_counts(tenant_id="default"):
    """Return indexed chunk counts by domain for this tenant only."""
    try:
        with db() as con:
            rows=con.execute("SELECT domain, COUNT(*) FROM chunk_fts WHERE tenant_id=? GROUP BY domain",(tenant_id,)).fetchall()
        return {str(k):int(v) for k,v in rows}
    except sqlite3.OperationalError:
        return {}
