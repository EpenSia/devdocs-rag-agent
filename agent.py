"""
3단계: 프레임워크 없이 직접 만드는 에이전트 루프 (ReAct)
---------------------------------------------------------
Claude Desktop이 대신 해 주던 일을 이 파일이 직접 한다.

  [agent.py]  ──(1) MCP 서버 실행·연결, 도구 목록 받기──▶  [mcp_server.py → Chroma]
      │
      ├─(2) 질문 + 도구 목록을 Claude API에 전달
      ├─(3) Claude가 tool_use를 요청하면 → MCP로 도구 실행 → 결과(tool_result)를 다시 전달
      └─(4) Claude가 더 이상 도구를 요청하지 않으면 → 최종 답변 출력

사용법:
  uv run agent.py "How do I run background tasks after returning a response?"
  uv run agent.py            # 대화 모드 (빈 줄 또는 exit 입력 시 종료)
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_TURNS = 6  # 무한 루프 방지: 도구 호출 왕복 최대 횟수

SYSTEM_PROMPT = """You are a FastAPI documentation assistant.
Use the tools to find evidence in the official docs before answering.
If the first search is not enough, search again with a different query or read the full page.
Cite the source file for each key point. Answer in Korean, keep code and API names in English."""


# ── 로그 출력 도우미 (루프 내부를 눈으로 보기 위함) ──────
def log(tag: str, msg: str = ""):
    print(f"\033[90m[{tag}]\033[0m {msg}", flush=True)


def short(text: str, n: int = 150) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " ..."


# ── MCP 도구 → Claude API 도구 형식으로 변환 ─────────────
# MCP와 Claude API는 도구를 표현하는 방식이 거의 같다 (이름 / 설명 / JSON 스키마).
# (mcp 2.x는 snake_case, 1.x는 camelCase 속성을 쓰므로 둘 다 지원)
def to_claude_tool(t) -> dict:
    schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema")
    return {"name": t.name, "description": t.description or "", "input_schema": schema}


def result_text(result) -> tuple[str, bool]:
    text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    is_error = bool(getattr(result, "is_error", None) or getattr(result, "isError", False))
    return text, is_error


# ── 핵심: 에이전트 루프 ───────────────────────────────────
async def run_agent(question: str, session: ClientSession, tools: list[dict], llm) -> str:
    messages = [{"role": "user", "content": question}]

    for turn in range(1, MAX_TURNS + 1):
        # (생각) Claude에게 지금까지의 대화 + 사용할 수 있는 도구 목록을 보낸다
        resp = llm.messages.create(
            model=MODEL, max_tokens=2048, system=SYSTEM_PROMPT,
            tools=tools, messages=messages,
        )
        log(f"turn {turn}", f"stop_reason = {resp.stop_reason}")

        # Claude의 응답(텍스트 + 도구 요청)을 대화 기록에 그대로 남긴다
        messages.append({"role": "assistant", "content": resp.content})

        # 도구 요청이 없으면 → 최종 답변
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        # (행동 + 관찰) Claude가 요청한 도구를 '내 코드가' 실제로 실행한다
        tool_results = []
        for block in resp.content:
            if block.type == "text" and block.text.strip():
                log("think", short(block.text))
            if block.type != "tool_use":
                continue
            log("tool_use", f"{block.name}({json.dumps(block.input, ensure_ascii=False)})")
            try:
                result = await session.call_tool(block.name, block.input)
                text, is_error = result_text(result)
            except Exception as e:  # 도구 실패도 Claude에게 알려서 스스로 대처하게 한다
                text, is_error = f"Tool error: {e}", True
            log("result", short(text))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,  # 어떤 요청에 대한 결과인지 짝을 맞춤
                "content": text,
                "is_error": is_error,
            })

        # 도구 결과는 'user' 역할로 돌려준다 → 다음 turn에서 Claude가 결과를 보고 재판단
        messages.append({"role": "user", "content": tool_results})

    return f"(최대 반복 횟수 {MAX_TURNS}회에 도달해 중단했습니다.)"


# ── MCP 서버 연결 + 실행 ──────────────────────────────────
async def main():
    # Claude Desktop 설정 파일의 command/args와 같은 역할:
    # 현재 가상환경의 파이썬으로 mcp_server.py를 하위 프로세스로 띄우고 stdio로 연결
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(BASE_DIR / "mcp_server.py")],
        cwd=str(BASE_DIR),
    )
    llm = Anthropic()  # ANTHROPIC_API_KEY 자동 사용

    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = [to_claude_tool(t) for t in (await session.list_tools()).tools]
        log("mcp", f"연결됨. 사용 가능한 도구: {[t['name'] for t in tools]}")

        if len(sys.argv) > 1:  # 질문 1개 모드
            print("\n" + await run_agent(" ".join(sys.argv[1:]), session, tools, llm))
            return

        while True:  # 대화 모드 (질문마다 새 대화로 시작)
            q = input("\n질문> ").strip()
            if q in {"", "exit", "quit"}:
                break
            print("\n" + await run_agent(q, session, tools, llm))


if __name__ == "__main__":
    asyncio.run(main())
