"""
1단계: 단일 파일로 돌아가는 기본 RAG
-----------------------------------------
흐름: 문서 로드 → 청킹 → 임베딩/저장(Chroma) → 검색 → 프롬프트 구성 → Claude 답변

사용법:
  uv run rag_basic.py index ./data/fastapi_docs          # 1) 문서 인덱싱
  uv run rag_basic.py search "background tasks"          # 2) 검색 결과만 확인 (LLM 호출 없음)
  uv run rag_basic.py ask "How do I run background tasks?"  # 3) RAG 답변

필요한 환경변수(.env):
  ANTHROPIC_API_KEY=sk-ant-...
  CLAUDE_MODEL=claude-sonnet-5-5   # 선택
"""

import hashlib
import os
import re
import sys
from pathlib import Path

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

# 이 파일이 있는 폴더(프로젝트 최상위)를 기준으로 경로를 잡음
# → 어느 위치에서 실행하든(예: Claude Desktop이 MCP 서버를 띄울 때) 같은 DB/.env를 찾음
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── 설정 ──────────────────────────────────────────────
DB_DIR = str(BASE_DIR / "chroma_db")  # 벡터 DB가 저장될 로컬 폴더
COLLECTION = "fastapi_docs"
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
CHUNK_SIZE = 1000                # 청크 최대 글자 수
CHUNK_OVERLAP = 150              # 청크 간 겹치는 글자 수 (문맥 끊김 방지)
TOP_K = 5                        # 검색해서 가져올 청크 개수


def get_collection():
    # 임베딩 함수를 지정하지 않으면 Chroma 기본 로컬 임베딩 모델(all-MiniLM-L6-v2)을 사용
    # → 별도 API 키 없이 임베딩 가능. 첫 실행 시 모델을 자동 다운로드함.
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )


# ── 1. 로드 ───────────────────────────────────────────
def load_markdown(folder: str):
    for path in sorted(Path(folder).rglob("*.md")):
        yield path, path.read_text(encoding="utf-8", errors="ignore")


# ── 2. 청킹 ───────────────────────────────────────────
def split_by_headers(text: str):
    """마크다운 헤더(#, ##, ###) 단위로 섹션 분리. 코드 블록 안의 '#'은 무시."""
    sections, title, buf, in_code = [], "(intro)", [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,3} ", line):
            if any(l.strip() for l in buf):
                sections.append((title, "\n".join(buf)))
            title, buf = line.lstrip("#").strip(), [line]
        else:
            buf.append(line)
    if any(l.strip() for l in buf):
        sections.append((title, "\n".join(buf)))
    return sections


def chunk_text(text: str):
    """섹션이 너무 길면 겹침(overlap)을 두고 고정 길이로 자름."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), step)]


# ── 3. 임베딩 + 저장 ──────────────────────────────────
def index(folder: str):
    col = get_collection()
    ids, docs, metas = [], [], []
    for path, text in load_markdown(folder):
        for sec_idx, (title, section) in enumerate(split_by_headers(text)):
            for i, chunk in enumerate(chunk_text(section)):
                # 결정적 ID: 같은 파일을 다시 인덱싱하면 덮어쓰기(upsert)됨
                # 섹션 제목이 아닌 '섹션 순번'을 써야 같은 제목(예: "Recap")이 여러 번 나와도 충돌하지 않음
                cid = hashlib.md5(f"{path}|{sec_idx}|{i}".encode()).hexdigest()
                ids.append(cid)
                docs.append(chunk)
                metas.append({"source": str(path), "section": title})

    batch = 200  # 한 번에 너무 많이 넣으면 느리거나 제한에 걸리므로 나눠서 저장
    for s in range(0, len(ids), batch):
        col.upsert(ids=ids[s : s + batch], documents=docs[s : s + batch],
                   metadatas=metas[s : s + batch])
        print(f"  저장 중... {min(s + batch, len(ids))}/{len(ids)}")
    print(f"✅ 인덱싱 완료: 청크 {len(ids)}개 (총 {col.count()}개)")


# ── 4. 검색 ───────────────────────────────────────────
def retrieve(query: str, k: int = TOP_K):
    res = get_collection().query(query_texts=[query], n_results=k)
    return [
        {"text": d, "source": m["source"], "section": m["section"], "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def search(query: str):
    for i, h in enumerate(retrieve(query), 1):
        print(f"\n[{i}] {h['source']} › {h['section']}  (distance={h['distance']:.3f})")
        print(h["text"][:300].replace("\n", " ") + " ...")


# ── 5. 생성 (프롬프트 구성 + LLM 호출) ────────────────
SYSTEM_PROMPT = """You are a technical assistant for FastAPI documentation.
Answer ONLY using the provided context. Cite sources like [1], [2].
If the context does not contain the answer, say you don't know.
Answer in Korean, but keep code and API names in English."""


def ask(query: str):
    hits = retrieve(query)
    context = "\n\n".join(
        f"[{i}] (source: {h['source']} › {h['section']})\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )
    client = Anthropic()  # ANTHROPIC_API_KEY 환경변수를 자동으로 읽음
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user",
                   "content": f"<context>\n{context}\n</context>\n\nQuestion: {query}"}],
    )
    print(resp.content[0].text)
    print("\n── 참고한 출처 ──")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['source']} › {h['section']}")


# ── CLI ───────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in {"index", "search", "ask"}:
        print(__doc__)
        sys.exit(1)
    cmd, arg = sys.argv[1], " ".join(sys.argv[2:])
    {"index": index, "search": search, "ask": ask}[cmd](arg)
