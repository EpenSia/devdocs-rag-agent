"""
7단계: 웹 데모 실행 환경 (화면과 분리된 부분)
------------------------------------------------
app.py(Streamlit 화면)가 사용하는 '뒤쪽' 기능만 모았다. 화면 없이도 테스트할 수 있다.

  ensure_index()   : 처음 실행 시 FastAPI 문서·예제 코드를 받아 인덱싱 (클라우드는 재시작마다 디스크가 비워짐)
  AgentRuntime     : MCP 서버 연결을 한 번만 열어 두고, 질문이 올 때마다 에이전트를 실행
  Quota            : 하루 전체 질문 수 제한 (공개 데모의 API 비용 보호)
  doc_url / extract_sources : 답변 근거를 FastAPI 공식 문서 링크와 GitHub 이슈 링크로 변환
"""

import asyncio
import concurrent.futures
import io
import os
import re
import subprocess
import sys
import tarfile
import threading
import urllib.request
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "fastapi_docs"
SRC_DIR = BASE_DIR / "data" / "docs_src"
READY_MARK = BASE_DIR / "chroma_db" / ".ready"
FASTAPI_REF = os.getenv("FASTAPI_REF", "master")  # 받을 FastAPI 저장소 버전 (브랜치나 태그)
DOCS_SITE = "https://fastapi.tiangolo.com/"


# ── 1. 처음 실행 시 데이터 준비 ─────────────────────────────
def _download_fastapi_docs(ref: str = FASTAPI_REF) -> None:
    """GitHub에서 FastAPI 저장소 압축 파일을 받아 문서(docs/en/docs)와 예제 코드(docs_src)만 꺼낸다."""
    url = f"https://codeload.github.com/fastapi/fastapi/tar.gz/{ref}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        root = tar.getmembers()[0].name.split("/")[0]  # 예: 'fastapi-master'
        targets = {f"{root}/docs/en/docs/": DOCS_DIR, f"{root}/docs_src/": SRC_DIR}
        for member in tar.getmembers():
            if not member.isfile():
                continue
            for prefix, dest in targets.items():
                if member.name.startswith(prefix):
                    out = (dest / member.name[len(prefix):]).resolve()
                    if not out.is_relative_to(dest.resolve()):  # '../' 같은 경로로 밖에 쓰는 것 방지
                        continue
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(tar.extractfile(member).read())


def ensure_index(log=print) -> None:
    """인덱스가 없으면 만든다. 인덱싱은 별도 프로세스에서 실행해 끝나면 메모리를 돌려받는다."""
    existing_db = (BASE_DIR / "chroma_db" / "chroma.sqlite3").exists() and DOCS_DIR.is_dir()
    if READY_MARK.exists() or existing_db:  # 로컬에서 이미 인덱싱해 둔 경우 그대로 사용
        return
    if not DOCS_DIR.is_dir() or not any(DOCS_DIR.rglob("*.md")):
        log("FastAPI 문서를 내려받는 중...")
        _download_fastapi_docs()
    log("문서를 인덱싱하는 중... (처음 한 번, 2~3분)")
    subprocess.run([sys.executable, str(BASE_DIR / "rag_basic.py"), "index", str(DOCS_DIR)],
                   cwd=BASE_DIR, check=True)
    READY_MARK.parent.mkdir(exist_ok=True)
    READY_MARK.write_text("ok", encoding="utf-8")


# ── 2. 에이전트 실행 환경 ─────────────────────────────────
class AgentRuntime:
    """백그라운드 스레드에 이벤트 루프를 하나 띄워 MCP 연결을 계속 열어 둔다.

    Streamlit은 화면을 그릴 때마다 스크립트를 처음부터 다시 실행하므로, 질문마다 MCP 서버를
    새로 띄우면 매번 수 초씩 걸린다. 그래서 연결은 이 객체 하나가 계속 들고 있고,
    질문은 run_coroutine_threadsafe로 그 루프에 넘긴다.
    """

    def __init__(self, max_concurrent: int = 2, timeout: float = 180):
        import agent  # 환경 변수(DEMO_MODE 등)가 정해진 뒤에 불러와야 함

        self.agent = agent
        self.max_concurrent = max_concurrent
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name="agent-loop").start()
        ready: concurrent.futures.Future = concurrent.futures.Future()
        asyncio.run_coroutine_threadsafe(self._serve(ready), self.loop)
        ready.result(timeout=timeout)  # 연결 실패 시 여기서 예외가 올라옴

    async def _serve(self, ready: concurrent.futures.Future) -> None:
        """MCP 연결을 '살아 있는 작업' 안에서 열어 둔다 (연결을 연 작업이 끝나면 정리 시 오류가 나기 때문)."""
        from contextlib import AsyncExitStack

        try:
            async with AsyncExitStack() as stack:
                self.registry, self.tools = await self.agent.connect_servers(stack)
                self.llm = self.agent.Anthropic()
                self.limit = asyncio.Semaphore(self.max_concurrent)  # 동시에 실행할 질문 수
                ready.set_result(True)
                await asyncio.Event().wait()  # 앱이 살아 있는 동안 계속 대기
        except BaseException as e:
            if not ready.done():
                ready.set_exception(e)
            raise

    @property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tools]

    def ask(self, question: str, on_event) -> concurrent.futures.Future:
        """질문을 에이전트에 넘긴다. 반환된 Future의 결과는 (답변, 실행 기록)."""

        async def job():
            async with self.limit:
                trace: dict = {}
                answer = await self.agent.run_agent(question, self.registry, self.tools, self.llm,
                                                    trace, on_event=on_event)
                return answer, trace

        return asyncio.run_coroutine_threadsafe(job(), self.loop)


# ── 3. 하루 전체 사용량 제한 ───────────────────────────────
class Quota:
    """하루 질문 수 제한. 메모리에만 저장하므로 앱이 재시작되면 초기화된다.
    진짜 비용 상한은 Claude Console의 지출 한도(spend limit)로 따로 건다."""

    def __init__(self, daily_limit: int):
        self.daily_limit = daily_limit
        self._lock = threading.Lock()
        self._day, self._used = date.today(), 0

    def _roll(self):
        if self._day != date.today():
            self._day, self._used = date.today(), 0

    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return max(self.daily_limit - self._used, 0)

    def take(self) -> bool:
        with self._lock:
            self._roll()
            if self._used >= self.daily_limit:
                return False
            self._used += 1
            return True


# ── 4. 근거 링크 만들기 ───────────────────────────────────
SOURCE_RE = re.compile(r"source:\s*(\S+\.md)")
ISSUE_RE = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/(?:issues|discussions)/\d+")


def doc_url(source: str) -> tuple[str, str]:
    """'data\\fastapi_docs\\tutorial\\cors.md' → ('tutorial/cors', 'https://fastapi.tiangolo.com/tutorial/cors/')"""
    path = source.replace("\\", "/")
    rel = path.split("fastapi_docs/", 1)[-1].removesuffix(".md")
    page = rel.removesuffix("index").rstrip("/")
    return (rel, DOCS_SITE + (page + "/" if page else ""))


def extract_sources(tool_results: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    """에이전트가 받은 도구 결과에서 문서 출처와 GitHub 이슈 링크를 순서대로, 중복 없이 뽑는다."""
    docs, issues = {}, {}
    for text in tool_results:
        for src in SOURCE_RE.findall(text):
            label, url = doc_url(src)
            docs.setdefault(url, label)
        for url in ISSUE_RE.findall(text):
            issues.setdefault(url, None)
    return [(label, url) for url, label in docs.items()], list(issues)
