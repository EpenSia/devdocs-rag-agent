"""
RAG 파이프라인 (1단계 → 5단계 고도화)
-----------------------------------------
흐름: 문서 로드 → [코드 참조 치환] → 청킹 → 임베딩/저장(Chroma) → [하이브리드 검색] → 답변

5단계에서 추가된 것:
  - 코드 참조 치환: 문서 속 {* ../../docs_src/xxx.py *} 지시문을 실제 예제 코드로 바꿈
    · 검색용 텍스트에는 코드를 빼고(설명 문장만), 반환용 텍스트(full_text)에만 코드를 넣음
      → 임베딩·BM25가 코드에 희석되지 않으면서, Claude는 실제 코드를 받음
  - 제목 정리: "Recap { #recap }" → "Recap"
  - 하이브리드 검색: 벡터 검색(의미) + BM25(키워드)를 RRF로 합침
  - 지연 import: chromadb/anthropic을 실제로 필요할 때 불러와 MCP 서버 시작 속도 개선

사용법:
  uv run rag_basic.py index ./data/fastapi_docs                # 인덱싱
  uv run rag_basic.py search "background tasks"                # 하이브리드 검색 (기본)
  uv run rag_basic.py search --mode=vector "background tasks"  # vector / bm25 / hybrid 비교
  uv run rag_basic.py ask "How do I run background tasks?"     # RAG 답변
"""

import hashlib
import math
import os
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── 설정 ──────────────────────────────────────────────
DB_DIR = str(BASE_DIR / "chroma_db")
DOCS_SRC_DIR = BASE_DIR / "data" / "docs_src"  # FastAPI 저장소의 예제 코드 폴더
COLLECTION = "fastapi_docs"
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5
MAX_RETURN_CHARS = 4000  # 반환용 청크 최대 길이 (검색 결과 5개가 Claude 컨텍스트를 과하게 차지하지 않게)
RRF_K = 60  # RRF 상수: 순위가 낮은 결과의 영향력을 완만하게 줄임 (관례적으로 60)
SEARCH_MODE = os.getenv("SEARCH_MODE", "hybrid")  # 6단계 평가 결과로 정함: vector / bm25 / hybrid


@lru_cache(maxsize=1)  # DB 연결은 한 번만 만들고 재사용
def get_collection():
    import chromadb  # 지연 import: 무거운 라이브러리를 실제로 쓸 때만 불러옴

    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


# ── 1. 로드 + 전처리 ───────────────────────────────────
# {* ../../docs_src/a/b.py hl[3] *}  또는  {!> ../../docs_src/a/b.py!}
INCLUDE_RE = re.compile(r"\{\*\s*(\S+)[^}]*?\*\}|\{!>?\s*(\S+?)\s*!\}")
LANG = {".py": "python", ".html": "html", ".css": "css", ".js": "javascript", ".toml": "toml"}


# 검색용 텍스트에서 코드 자리를 표시하는 '한 글자' 마커 (유니코드 사용자 정의 영역).
# 한 글자라서 청크를 자를 때 마커가 반으로 잘릴 일이 없다.
MARKER_BASE = 0xE000
MARKER_RE = re.compile("[-]")


def expand_includes(text: str) -> tuple[str, dict[str, str]]:
    """코드 참조 지시문을 마커 한 글자로 바꾸고, 마커 → 실제 코드 블록 사전을 함께 반환한다.

    검색용: 마커만 지우면 설명 문장만 남음 (임베딩·BM25가 코드에 희석되지 않음)
    반환용: 마커를 실제 코드로 바꾸면 예제 코드가 포함된 원래 문서가 됨
    """
    codes: dict[str, str] = {}

    def repl(m):
        ref = m.group(1) or m.group(2)
        pos = ref.find("docs_src/")
        path = DOCS_SRC_DIR / ref[pos + len("docs_src/"):] if pos >= 0 else None
        if path is None or not path.is_file():
            return m.group(0)  # 못 찾으면 원문 유지
        marker = chr(MARKER_BASE + len(codes))
        code = path.read_text(encoding="utf-8", errors="ignore").strip()
        codes[marker] = f"\n```{LANG.get(path.suffix, '')}\n{code}\n```\n"
        return marker

    return INCLUDE_RE.sub(repl, text), codes


def render_doc(text: str) -> str:
    """문서 전체를 사람이 읽는 형태로: 코드 참조를 실제 코드로 바꿔서 반환 (read_doc용)."""
    text, codes = expand_includes(text)
    return MARKER_RE.sub(lambda m: codes.get(m.group(0), ""), text)


def load_markdown(folder: str):
    for path in sorted(Path(folder).rglob("*.md")):
        if path.name.startswith("_"):  # 내부용 파일 제외 (예: _llm-test.md = 번역 도구 테스트 문서)
            continue
        yield path, path.read_text(encoding="utf-8", errors="ignore")


# ── 2. 청킹 ───────────────────────────────────────────
def clean_title(title: str) -> str:
    return re.sub(r"\s*\{\s*#[^}]*\}\s*$", "", title).strip()


def split_by_headers(text: str):
    """마크다운 헤더(#, ##, ###) 단위로 섹션 분리. 코드 블록 안의 '#'은 무시."""
    sections, title, buf, in_code = [], "(intro)", [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,3} ", line):
            if any(l.strip() for l in buf):
                sections.append((title, "\n".join(buf)))
            title, buf = clean_title(line.lstrip("#")), [line]
        else:
            buf.append(line)
    if any(l.strip() for l in buf):
        sections.append((title, "\n".join(buf)))
    return sections


def chunk_text(text: str):
    if len(text) <= CHUNK_SIZE:
        return [text]
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), step)]


def build_chunks(folder: str):
    """문서 폴더 → (ids, 검색용 텍스트, 메타데이터, 치환 수). DB와 무관해서 단독 테스트 가능.

    - documents(검색용): 설명 문장만. Chroma 임베딩과 BM25가 이 텍스트로 계산됨
    - metadata["full_text"](반환용): 같은 청크에 실제 예제 코드를 끼워 넣은 텍스트
    """
    ids, docs, metas, n_inc = [], [], [], 0
    for path, text in load_markdown(folder):
        text, codes = expand_includes(text)
        n_inc += len(codes)
        try:  # 출처는 프로젝트 기준 상대경로로 저장 (read_doc과 일치)
            source = str(path.resolve().relative_to(BASE_DIR))
        except ValueError:
            source = str(path)
        for sec_idx, (title, section) in enumerate(split_by_headers(text)):
            # 청크 길이는 설명 문장 기준으로 잰다 (코드 길이 때문에 설명이 잘게 쪼개지지 않게)
            for i, chunk in enumerate(chunk_text(section)):
                search_text = MARKER_RE.sub("", chunk).strip()
                full_text = MARKER_RE.sub(lambda m: codes.get(m.group(0), ""), chunk)
                if len(full_text) > MAX_RETURN_CHARS:  # 아주 긴 예제 코드가 붙은 청크(약 0.6%)
                    full_text = full_text[:MAX_RETURN_CHARS] + "\n...(truncated: use read_doc for the full page)"
                ids.append(hashlib.md5(f"{source}|{sec_idx}|{i}".encode()).hexdigest())
                docs.append(search_text or title)
                metas.append({"source": source, "section": title, "full_text": full_text})
    return ids, docs, metas, n_inc


# ── 3. 임베딩 + 저장 ──────────────────────────────────
def index(folder: str):
    if not DOCS_SRC_DIR.is_dir():
        print(f"⚠️  예제 코드 폴더가 없습니다: {DOCS_SRC_DIR} (코드 참조 치환 없이 진행)")
    ids, docs, metas, n_inc = build_chunks(folder)
    col = get_collection()
    batch = 200
    for s in range(0, len(ids), batch):
        col.upsert(ids=ids[s : s + batch], documents=docs[s : s + batch],
                   metadatas=metas[s : s + batch])
        print(f"  저장 중... {min(s + batch, len(ids))}/{len(ids)}")
    _bm25_index.cache_clear()
    print(f"✅ 인덱싱 완료: 청크 {len(ids)}개, 코드 참조 {n_inc}개 연결 (DB 총 {col.count()}개)")


# ── 4. 검색 ───────────────────────────────────────────
# 4-1. 벡터 검색: 의미가 비슷한 문장을 찾음 ("작업을 응답 후에" ≈ "background tasks")
def _vector_search(query: str, k: int):
    res = get_collection().query(query_texts=[query], n_results=k)
    return [
        # 검색은 설명 문장(d)으로, 반환은 코드가 들어간 full_text로 (예전 인덱스면 d 그대로)
        {"id": i, "text": m.get("full_text", d), "source": m["source"], "section": m["section"],
         "distance": dist}
        for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                 res["metadatas"][0], res["distances"][0])
    ]


# 4-2. BM25: 단어가 정확히 겹치는 문서를 찾음 (StreamingResponse, CORSMiddleware 같은 API 이름에 강함)
TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = {"a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "is", "are", "be",
             "with", "how", "do", "i", "my", "can", "what", "when", "it", "from", "by", "as"}


def tokenize(text: str):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """BM25 직접 구현 (라이브러리 없이 20줄). 점수 = 단어의 희귀도(IDF) × 문서 안 빈도(TF, 길이 보정)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for c in self.tf for t in c)  # 각 단어가 등장한 문서 수
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            scores.append(s)
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return [(i, scores[i]) for i in ranked[:k] if scores[i] > 0]


@lru_cache(maxsize=1)
def _bm25_index():
    """DB에 저장된 청크를 한 번 읽어 BM25 색인을 메모리에 만든다."""
    data = get_collection().get(include=["documents", "metadatas"])
    return BM25(data["documents"]), data


def _bm25_search(query: str, k: int):
    bm25, data = _bm25_index()
    return [
        {"id": data["ids"][i], "text": data["metadatas"][i].get("full_text", data["documents"][i]),
         "source": data["metadatas"][i]["source"],
         "section": data["metadatas"][i]["section"], "distance": None}
        for i, _ in bm25.top(query, k)
    ]


# 4-3. 하이브리드: 두 결과의 '순위'를 합침 (RRF). 점수 단위가 달라도 순위는 비교 가능하기 때문.
def retrieve(query: str, k: int = TOP_K, mode: str | None = None):
    mode = mode or SEARCH_MODE
    pool = max(k * 4, 20)  # 각 방식에서 넉넉히 후보를 뽑은 뒤 합침
    if mode == "vector":
        return _vector_search(query, k)
    if mode == "bm25":
        return _bm25_search(query, k)

    vec, kw = _vector_search(query, pool), _bm25_search(query, pool)
    scores, hits = {}, {}
    for results, tag in ((vec, "vector"), (kw, "bm25")):
        for rank, h in enumerate(results):
            scores[h["id"]] = scores.get(h["id"], 0) + 1 / (RRF_K + rank + 1)
            hit = hits.setdefault(h["id"], {**h, "matched_by": []})
            hit["matched_by"].append(tag)
            if h["distance"] is not None:
                hit["distance"] = h["distance"]
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [hits[i] for i in best]


def search(query: str, mode: str | None = None):
    for i, h in enumerate(retrieve(query, mode=mode), 1):
        dist = f"distance={h['distance']:.3f}" if h.get("distance") is not None else "keyword only"
        via = "+".join(h.get("matched_by", [mode or SEARCH_MODE]))
        print(f"\n[{i}] {h['source']} › {h['section']}  ({dist}, via {via})")
        print(h["text"][:300].replace("\n", " ") + " ...")


# ── 5. 생성 ───────────────────────────────────────────
SYSTEM_PROMPT = """You are a technical assistant for FastAPI documentation.
Answer ONLY using the provided context. Cite sources like [1], [2].
If the context does not contain the answer, say you don't know.
Answer in Korean, but keep code and API names in English."""


def ask(query: str):
    from anthropic import Anthropic  # 지연 import

    hits = retrieve(query)
    context = "\n\n".join(
        f"[{i}] (source: {h['source']} › {h['section']})\n{h['text']}" for i, h in enumerate(hits, 1)
    )
    resp = Anthropic().messages.create(
        model=MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"<context>\n{context}\n</context>\n\nQuestion: {query}"}],
    )
    print(resp.content[0].text)
    print("\n── 참고한 출처 ──")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['source']} › {h['section']}")


# ── CLI ───────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]
    mode = next((a.split("=", 1)[1] for a in args if a.startswith("--mode=")), None)
    args = [a for a in args if not a.startswith("--mode=")]
    if len(args) < 2 or args[0] not in {"index", "search", "ask"}:
        print(__doc__)
        sys.exit(1)
    cmd, arg = args[0], " ".join(args[1:])
    if cmd == "search":
        search(arg, mode)
    else:
        {"index": index, "ask": ask}[cmd](arg)
