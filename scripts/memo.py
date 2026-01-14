# check the first 10 chunks from the ingest_pdfs.py output
python3 - << 'PY'
import json
from pathlib import Path

p = Path("index/chunks.jsonl")
for i, line in enumerate(p.open(encoding="utf-8")):
    if i >= 10: break
    r = json.loads(line)
    print("==", i, r["chunk_id"], r["meta"]["topic"], "page", r["meta"]["page"])
    print(r["text"][:200].replace("\n"," "), "...\n")
PY


# check the skip ratio
python3 - << 'PY'
import json
from pathlib import Path

p = Path("index/chunks.jsonl")
n = 0
skip = 0
for line in p.open(encoding="utf-8"):
    r = json.loads(line)
    n += 1
    if r["meta"].get("skip"):
        skip += 1
print("chunks:", n, "skip:", skip, "skip_ratio:", skip/n if n else 0)
PY

# set up OpenAI API key
pip install openai
export OPENAI_API_KEY="...あなたの鍵..."

# call api with RAG
python3 rag/answerer_api.py \
  --question "Explain strict exogeneity and why it matters for FE vs FD." \
  --topic econometrics \
  --top_k 8 \
  --temperature 0.2
