import argparse, hashlib, re, uuid, io
from pathlib import Path
import fitz, pytesseract
from PIL import Image
from . import config
from .pii import mask_pii
from .index import embedder, index_chunks, delete_document
from .lexical import replace_document

def extract_pdf(path):
    doc=fitz.open(path); pages=[]
    for pno,page in enumerate(doc,1):
        native=page.get_text("text",sort=True).strip()
        method="native"
        if len(native)<50:
            pix=page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
            img=Image.frombytes("RGB",(pix.width,pix.height),pix.samples)
            native=pytesseract.image_to_string(img,lang=config.OCR_LANG)
            method="page_ocr"
        # Extract and OCR embedded raster images not represented in page text.
        image_text=[]
        for info in page.get_images(full=True):
            try:
                raw=doc.extract_image(info[0])["image"]
                im=Image.open(io.BytesIO(raw)).convert("RGB")
                if im.width*im.height>40000:
                    t=pytesseract.image_to_string(im,lang=config.OCR_LANG).strip()
                    if t and t not in native: image_text.append(t)
            except Exception: pass
        # Extract native tables with PyMuPDF where supported.
        tables=[]
        try:
            finder=page.find_tables()
            for table in finder.tables:
                rows=table.extract()
                if rows:
                    tables.append("\n".join(" | ".join(str(c or "") for c in row) for row in rows))
        except Exception: pass
        combined=native
        if tables: combined += "\n\n[TABLE DATA]\n" + "\n\n".join(tables)
        if image_text: combined += "\n\n[IMAGE OCR]\n" + "\n\n".join(image_text)
        combined=re.sub(r"[ \t]+"," ",combined)
        combined=re.sub(r"\n{3,}","\n\n",combined).strip()
        if combined:
            safe, pii_count=mask_pii(combined)
            pages.append({"page":pno,"text":safe,"bbox":list(page.rect),
                          "extraction":method,"pii_entities_masked":pii_count})
    doc.close()
    return pages

def clean_repeated_headers(pages):
    if len(pages)<4:return pages
    counts={}
    for p in pages:
        lines=[x.strip() for x in p["text"].splitlines() if x.strip()]
        for line in set(lines[:2]+lines[-2:]):
            if 4<len(line)<120: counts[line]=counts.get(line,0)+1
    repeated={x for x,n in counts.items() if n/len(pages)>=.65}
    for p in pages:
        ls=p["text"].splitlines()
        p["text"]="\n".join(x for i,x in enumerate(ls)
          if x.strip() not in repeated or (i>=2 and i<len(ls)-2))
    return pages

def chunks(text, tokenizer):
    ids=tokenizer.encode(text,add_special_tokens=False)
    step=max(1,config.CHUNK_TOKENS-config.CHUNK_OVERLAP)
    for start in range(0,len(ids),step):
        piece=ids[start:start+config.CHUNK_TOKENS]
        if piece: yield tokenizer.decode(piece,skip_special_tokens=True)
        if start+config.CHUNK_TOKENS>=len(ids): break

def process(path, tenant_id="default", metadata=None):
    metadata=metadata or {}
    doc_id=hashlib.sha256(tenant_id.encode()+b":"+path.read_bytes()).hexdigest()[:24]
    pages=clean_repeated_headers(extract_pdf(path))
    tok=embedder().tokenizer; result=[]
    for p in pages:
        for i,part in enumerate(chunks(p["text"],tok)):
            cid=f"{doc_id}:{p['page']}:{i}"
            result.append({
              "id":str(uuid.uuid5(uuid.NAMESPACE_URL,cid)),"chunk_id":cid,
              "document_id":doc_id,"filename":path.name,"page":p["page"],
              "bbox":p["bbox"],"extraction":p["extraction"],
              "pii_entities_masked":p["pii_entities_masked"],
              "chunk_type":"text_or_table","tenant_id":tenant_id,
              "document_type":metadata.get("document_type","general"),
              "domain":metadata.get("domain","general"),
              "tags":metadata.get("tags",[]),"source":metadata.get("source","upload"),
              "created_at":metadata.get("created_at"),"text":part
            })
    return doc_id,result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dir",default=config.PDF_DIR); ap.add_argument("--file")
    args=ap.parse_args()
    paths=[Path(args.file)] if args.file else sorted(Path(args.dir).glob("*.pdf"))
    if not paths: raise SystemExit("No PDFs found")
    for path in paths:
        print("Processing",path.name)
        doc_id,rows=process(path)
        delete_document(doc_id)
        index_chunks(rows)
        replace_document(doc_id,rows)
        print("Indexed",len(rows),"chunks")
if __name__=="__main__":main()
