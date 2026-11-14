"""
4단계: 여러 MCP 서버를 쓰는 에이전트 + 사람 승인(Human-in-the-loop)
-------------------------------------------------------------------
3단계 루프는 그대로 두고, 연결하는 MCP 서버를 1개 → 2개로 늘렸다.

  [agent.py] ─ stdio ─▶ fastapi-docs (내가 만든 서버: search_docs, read_doc)
             └ HTTP  ─▶ github       (GitHub 공식 원격 서버: search_issues, issue_read, issue_write)

흐름: 문서 검색 → 부족하면 GitHub 이슈 검색 → 그래도 미해결이면 이슈 초안 작성
      → 쓰기 도구는 '사람이 승인해야만' 실행, 그리고 내 테스트 저장소에만 허용

.env 설정:
  ANTHROPIC_API_KEY=sk-ant-...
  GITHUB_TOKEN=github_pat_...                    # 없으면 문서 검색만으로 동작 (3단계와 동일)
  GITHUB_SANDBOX_REPO=내아이디/agent-sandbox      # 없으면 GitHub 읽기 전용 모드

사용법:
  uv run agent.py "질문"
  uv run agent.py            # 대화 모드
"""

import asyncio
import json
import os
import re
import sys
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_TURNS = 8
MAX_TOKENS = 4096  # 답변 최대 길이 (6단계: 2048에서는 read_doc 후 긴 답변이 중간에 잘림)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MCP_URL = os.getenv("GITHUB_MCP_URL", "https://api.githubcopilot.com/mcp/")
SANDBOX_REPO = os.getenv("GITHUB_SANDBOX_REPO", "").strip().lower()  # 예: "myid/agent-sandbox"

# GitHub 서버는 도구가 수십 개라, 필요한 것만 노출한다 (도구가 많을수록 LLM이 헷갈리고 토큰도 낭비)
GITHUB_TOOLS = ["search_issues", "issue_read", "issue_write"]
# 외부 세계를 바꾸는 도구 → 실행 전에 반드시 사람 승인
WRITE_TOOLS = {"issue_write"}

OUTSIDE_DOCS_TITLE = "문서 외 참고"  # 검색 결과에 없는 일반 지식을 모아 두는 섹션 제목 (평가 스크립트도 사용)

SYSTEM_PROMPT = f"""You are a FastAPI support agent.
Workflow:
1. Search the official docs first (search_docs, read_doc).
2. If the docs don't solve the problem, search existing GitHub issues/discussions
   with search_issues, e.g. query "repo:fastapi/fastapi is:issue <keywords>".
3. Only if still unresolved AND the user asks for it, draft an issue with issue_write.
   Issues may ONLY be created in the sandbox repo: {SANDBOX_REPO or "(not configured - do not create issues)"}.
   A human will review and approve the draft before it is created.

Grounding rules (6단계: 근거 밖 보충을 줄이기 위해 추가):
- The main answer must contain ONLY information found in tool results.
  Cite the source (doc file or issue URL) for each key point.
- If a result is cut off ("...(truncated") or lacks a detail you need (exact parameters,
  a full example, required steps), call read_doc on that source BEFORE answering.
  Do not fill gaps from memory.
- Helpful information that is NOT in the tool results (install commands, status codes,
  general Python/HTTP knowledge, tips) may appear ONLY in a final section titled
  "## {OUTSIDE_DOCS_TITLE}", kept short. If there is nothing to add, omit that section entirely.
- If the docs do not cover the question, say so plainly.

Language: ALWAYS write the entire answer in Korean (한국어), even when the question and
the docs are in English. Keep code, API names and file paths in English."""


def log(tag: str, msg: str = ""):
    if os.getenv("AGENT_QUIET"):  # 평가 스크립트에서 대량 실행할 때 로그 끄기
        return
    print(f"\033[90m[{tag}]\033[0m {msg}", flush=True)


def short(text: str, n: int = 150) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " ..."


def is_korean(text: str) -> bool:
    """한국어 답변인가. 한국어 문장에는 일본어 가나가 나오지 않고 한자도 거의 쓰지 않는다는 점을 이용.
    (6단계: 한자 비율만 보던 첫 버전은 첫 문장이 일본어인 답변을 놓쳤음)"""
    hangul = len(re.findall(r"[가-힣]", text))
    kana = len(re.findall(r"[぀-ヿ]", text))
    han = len(re.findall(r"[一-鿿]", text))
    return hangul > 0 and kana == 0 and han < 10  # 가나는 1자도 허용 안 함 (6단계: 'テス트'처럼 2자만 섞인 경우를 놓쳤음)


REWRITE_IN_KOREAN = ("Your answer above is not entirely in Korean. Rewrite the whole answer in Korean (한국어), "
                     "keeping the same content, code and sources. Do not call any tools.")


def to_claude_tool(t) -> dict:
    schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema")
    return {"name": t.name, "description": t.description or "", "input_schema": schema}


def result_text(result) -> tuple[str, bool]:
    text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    is_error = bool(getattr(result, "is_error", None) or getattr(result, "isError", False))
    return text, is_error


def compact_issues(text: str, body_chars: int = 300) -> str:
    """search_issues의 원본 JSON(이슈 본문 전체 포함)을 필요한 필드만 남겨 줄인다 → 토큰 절약."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    items = [
        {
            "number": it.get("number"), "title": it.get("title"), "state": it.get("state"),
            "url": it.get("html_url"), "comments": it.get("comments"),
            "updated_at": (it.get("updated_at") or "")[:10],
            "body": " ".join((it.get("body") or "").split())[:body_chars],
        }
        for it in data.get("items", [])
    ]
    return json.dumps({"total_count": data.get("total_count"), "items": items}, ensure_ascii=False)


# ── HTTP(원격) MCP 서버 연결 ───────────────────────────────
# stdio는 내 PC에서 서버 프로세스를 직접 띄우고, HTTP는 이미 떠 있는 원격 서버에 접속한다.
# (mcp 2.x / 1.x의 함수 이름이 달라서 둘 다 지원)
@asynccontextmanager
async def open_http(url: str, headers: dict):
    import mcp.client.streamable_http as sh

    if hasattr(sh, "streamable_http_client"):  # mcp 2.x
        async with sh.create_mcp_http_client(headers=headers) as http, \
                sh.streamable_http_client(url, http_client=http) as (read, write):
            yield read, write
    else:  # mcp 1.x
        async with sh.streamablehttp_client(url, headers=headers) as (read, write, _):
            yield read, write


# ── 사람 승인 + 안전 규칙 ─────────────────────────────────
def check_write(name: str, args: dict) -> str | None:
    """쓰기 도구 실행 전 검사. 막아야 하면 이유(str)를, 통과하면 None을 반환."""
    target = f"{args.get('owner', '')}/{args.get('repo', '')}".lower()
    if not SANDBOX_REPO:
        return "Write tools are disabled (GITHUB_SANDBOX_REPO not set)."
    if target != SANDBOX_REPO:  # 규칙 1: 내 테스트 저장소 외에는 절대 쓰지 않음 (사람에게 묻지도 않음)
        return f"Blocked: writing to '{target}' is not allowed. Only '{SANDBOX_REPO}'."

    # 규칙 2: 사람이 내용을 보고 승인해야만 실행
    print("\n" + "=" * 60)
    print(f"🛑 승인 요청: {name}  →  {target}")
    print(f"제목: {args.get('title', '')}")
    print("-" * 60)
    print(args.get("body", ""))
    print("=" * 60)
    if input("이대로 실행할까요? [y/N] ").strip().lower() != "y":
        return "The user rejected this action. Do not retry; tell the user the draft instead."
    return None


# ── 에이전트 루프 (3단계와 동일 + 도구 라우팅 + 승인) ─────────
async def run_agent(question: str, registry: dict, tools: list[dict], llm,
                    trace: dict | None = None) -> str:
    """trace에 dict를 넘기면 실행 기록(턴 수, 도구 호출, 도구 결과, 토큰, 시간)을 채워 준다 (6단계 평가용)."""
    messages = [{"role": "user", "content": question}]
    t = trace if trace is not None else {}
    t.update(turns=0, tool_calls=[], tool_results=[], input_tokens=0, output_tokens=0,
             truncated=False, language_fixed=False)
    started = time.perf_counter()

    for turn in range(1, MAX_TURNS + 1):
        resp = llm.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
            tools=tools, messages=messages,
        )
        log(f"turn {turn}", f"stop_reason = {resp.stop_reason}")
        messages.append({"role": "assistant", "content": resp.content})
        usage = getattr(resp, "usage", None)
        t["turns"] = turn
        t["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        t["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        t["seconds"] = round(time.perf_counter() - started, 1)

        if resp.stop_reason != "tool_use":
            answer = "".join(b.text for b in resp.content if b.type == "text")
            # 언어 가드레일: 프롬프트로 지시해도 가끔 중국어·일본어가 섞임 → 코드로 검사하고 한 번만 다시 쓰게 함
            if not is_korean(answer) and not t["language_fixed"] and turn < MAX_TURNS:
                log("warn", "답변이 한국어가 아니어서 다시 쓰도록 요청합니다")
                t["language_fixed"] = True
                messages.append({"role": "user", "content": REWRITE_IN_KOREAN})
                continue
            t["truncated"] = resp.stop_reason == "max_tokens"
            if t["truncated"]:  # 잘린 답변을 완성된 답변처럼 내보내지 않도록 표시
                log("warn", f"답변이 최대 길이({MAX_TOKENS} 토큰)에서 잘렸습니다")
                answer += "\n\n(⚠️ 답변이 길이 제한으로 중간에 잘렸습니다.)"
            return answer

        tool_results = []
        for block in resp.content:
            if block.type == "text" and block.text.strip():
                log("think", short(block.text))
            if block.type != "tool_use":
                continue

            server, session = registry.get(block.name, ("?", None))  # 이 도구를 가진 서버로 보낸다
            log("tool_use", f"{server}.{block.name}({short(json.dumps(block.input, ensure_ascii=False), 200)})")

            if session is None:  # LLM이 없는 도구를 지어내 요청한 경우
                blocked = f"Unknown tool '{block.name}'. Available: {list(registry)}"
            elif block.name in WRITE_TOOLS:
                blocked = check_write(block.name, block.input)
            else:
                blocked = None

            if blocked:
                text, is_error = blocked, True
            else:
                try:
                    result = await session.call_tool(block.name, block.input)
                    text, is_error = result_text(result)
                    if block.name == "search_issues" and not is_error:
                        before = len(text)
                        text = compact_issues(text)
                        log("compact", f"{before:,}자 → {len(text):,}자")
                    # 승인은 내 코드에서만 일어나므로, Claude가 알 수 있게 결과에 적어 준다
                    if block.name in WRITE_TOOLS and not is_error:
                        text = "[Approved by the user and executed]\n" + text
                except Exception as e:
                    text, is_error = f"Tool error: {e}", True

            log("result", ("❌ " if is_error else "") + short(text))
            t["tool_calls"].append({"name": block.name, "input": block.input})
            t["tool_results"].append(text)
            tool_results.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": text, "is_error": is_error,
            })

        messages.append({"role": "user", "content": tool_results})

    return f"(최대 반복 횟수 {MAX_TURNS}회에 도달해 중단했습니다.)"


# ── 서버 연결 → 도구 등록부 만들기 ───────────────────────────
async def connect_servers(stack: AsyncExitStack) -> tuple[dict, list[dict]]:
    registry: dict[str, tuple[str, ClientSession]] = {}  # 도구 이름 → (서버 이름, 세션)
    tools: list[dict] = []

    async def add(name: str, read, write, allow: list[str] | None = None):
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        for t in (await session.list_tools()).tools:
            if allow and t.name not in allow:
                continue
            if t.name in registry:  # 서로 다른 서버에 같은 이름의 도구가 있으면 충돌
                raise RuntimeError(f"Duplicate tool name: {t.name}")
            registry[t.name] = (name, session)
            tools.append(to_claude_tool(t))
        log("mcp", f"{name} 연결됨: {[n for n, (s, _) in registry.items() if s == name]}")

    # 서버 1: 내가 만든 문서 서버 (stdio)
    docs = StdioServerParameters(
        command=sys.executable, args=[str(BASE_DIR / "mcp_server.py")], cwd=str(BASE_DIR)
    )
    await add("fastapi-docs", *await stack.enter_async_context(stdio_client(docs)))

    # 서버 2: GitHub 공식 원격 서버 (HTTP)
    if GITHUB_TOKEN:
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "X-MCP-Toolsets": "issues",
            "X-MCP-Tools": ",".join(GITHUB_TOOLS),  # 서버 쪽에서도 도구 제한
        }
        if not SANDBOX_REPO:
            headers["X-MCP-Readonly"] = "true"  # 쓰기 대상이 없으면 읽기 도구만 받음
        read, write = await stack.enter_async_context(open_http(GITHUB_MCP_URL, headers))
        allow = GITHUB_TOOLS if SANDBOX_REPO else [t for t in GITHUB_TOOLS if t not in WRITE_TOOLS]
        await add("github", read, write, allow=allow)  # 클라이언트 쪽에서도 한 번 더 거름
        if not SANDBOX_REPO:
            log("mcp", "GITHUB_SANDBOX_REPO 미설정 → GitHub 읽기 전용 모드")
    else:
        log("mcp", "GITHUB_TOKEN 미설정 → 문서 검색만 사용")

    return registry, tools


async def main():
    llm = Anthropic()
    async with AsyncExitStack() as stack:
        registry, tools = await connect_servers(stack)

        if len(sys.argv) > 1:
            print("\n" + await run_agent(" ".join(sys.argv[1:]), registry, tools, llm))
            return

        while True:
            q = input("\n질문> ").strip()
            if q in {"", "exit", "quit"}:
                break
            print("\n" + await run_agent(q, registry, tools, llm))


if __name__ == "__main__":
    asyncio.run(main())
