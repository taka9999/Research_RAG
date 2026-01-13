from pathlib import Path

base = Path("research_rag")

dirs = [
    base / "data" / "papers",
    base / "data" / "notes",
    base / "data" / "regulations",
    base / "index",
    base / "rag",
    base / "eval",
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

print("research_rag directory structure created.")