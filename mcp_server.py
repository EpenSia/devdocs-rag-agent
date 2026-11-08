"""
2단계: 1단계 RAG 검색 기능을 MCP 서버로 감싸기
-----------------------------------------------
RAG 로직(rag_basic.py)은 한 줄도 바꾸지 않고, 그 위에 '표준 인터페이스'만 씌운다.
→ Claude Desktop, MCP Inspector 등 MCP를 지원하는 어떤 클라이언트든 이 기능을 도구로 쓸 수 있다.

이 서버가 노출하는 것:
  - Tool     search_docs(query, top_k)  : 벡터 검색 (LLM이 '호출'하는 함수)
  - Tool     read_doc(source)           : 검색된 문서의 전체 원문 읽기
  - Resource docs://stats               : 인덱스 상태 정보 (사람/앱이 '읽는' 데이터)

실행 / 테스트:
  uv run mcp dev mcp_server.py      # MCP Inspector(브라우저)로 테스트 (Node.js 필요)
  uv run mcp_server.py              # stdio 서버로 직접 실행 (보통은 클라이언트가 대신 실행함)

⚠️ stdio 방식은 표준출력(stdout)으로 JSON-RPC 메시지를 주고받는다.
   이 파일 안에서 print()를 쓰면 통신이 깨질 수 있으니, 디버깅 출력은 stderr로 보낼 것:
   print("debug", file=sys.stderr)
"""

from pathlib import Path

try:
    from mcp.server.mcpserver import MCPServer  # mcp 2.x (현재 버전)
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer  # mcp 1.x (구버전 호환)

from rag_basic import BASE_DIR, MODEL, get_collection, retrieve

DOCS_DIR = (BASE_DIR / "data" / "fastapi_docs").resolve()
MAX_DOC_CHARS = 20_000  # 문서가 너무 길면 LLM 컨텍스트를 낭비하므로 잘라서 반환

mcp = MCPServer(
    "fastapi-docs",
    instructions=(
        "Search and read the official FastAPI documentation. "
        "Call search_docs first, then read_doc only if a chunk is not enough."
    ),
)


# ── Tool 1: 검색 ──────────────────────────────────────
# 함수 이름 = 도구 이름, 타입 힌트 = 입력 스키마, docstring = 도구 설명.
# LLM은 이 docstring만 보고 '언제, 어떻게' 이 도구를 쓸지 판단한다 → 설명을 구체적으로!
@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> str:
    """Semantic search over the official FastAPI documentation.

    Use this for any question about FastAPI features, APIs, or usage.
    Write the query in English, since the docs are in English.
    Returns the most relevant text chunks with their `source` file and section.
    Lower distance means more relevant; above ~0.6 is usually unrelated.

    Args:
        query: What to look for, e.g. "run background tasks after response".
        top_k: Number of chunks to return (1-10).
    """
    top_k = max(1, min(top_k, 10))
    hits = retrieve(query, top_k)
    if not hits:
        return "No results found."
    return "\n\n---\n\n".join(
        f"[{i}] source: {h['source']}\n"
        f"section: {h['section']}\n"
        f"distance: {h['distance']:.3f}\n\n"
        f"{h['text']}"
        for i, h in enumerate(hits, 1)
    )


# ── Tool 2: 원문 읽기 ─────────────────────────────────
@mcp.tool()
def read_doc(source: str) -> str:
    """Read the full markdown of one documentation page.

    Use only when a chunk from search_docs is cut off or lacks context.
    Pass the `source` value exactly as returned by search_docs.
    """
    path = (BASE_DIR / source).resolve()
    # 보안: LLM이 넘긴 경로로 아무 파일이나 읽지 못하도록 문서 폴더 안의 .md만 허용
    if not path.is_relative_to(DOCS_DIR) or path.suffix != ".md" or not path.is_file():
        return f"Error: '{source}' is not a valid documentation file."
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS] + "\n\n...(truncated)"
    return text


# ── Resource: 읽기 전용 데이터 ─────────────────────────
# Tool은 LLM이 스스로 '호출'하는 것, Resource는 클라이언트(사람)가 골라서 '첨부'하는 데이터.
@mcp.resource("docs://stats")
def stats() -> str:
    """Index status: number of chunks and where the docs come from."""
    return (
        f"collection: {get_collection().name}\n"
        f"chunks: {get_collection().count()}\n"
        f"docs folder: {DOCS_DIR}\n"
        f"answer model (rag_basic ask): {MODEL}"
    )


if __name__ == "__main__":
    mcp.run()  # 기본 transport = "stdio"
