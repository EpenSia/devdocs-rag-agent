"""
7단계: 포트폴리오 시연용 웹 UI (Streamlit)
------------------------------------------
담당자가 링크 하나로 에이전트를 직접 써 보게 하는 화면.
답변만이 아니라 '에이전트가 어떤 도구를 왜 불렀는지'를 실시간으로 보여 주는 것이 핵심.

로컬 실행:  uv run streamlit run app.py
배포:       Streamlit Community Cloud (README의 '웹 데모 배포' 참고)

공개 데모 안전장치:
  - DEMO_MODE: GitHub는 읽기 전용, 이슈 작성은 승인 카드까지만 (실제 등록 없음)
  - 세션당 / 하루 전체 질문 수 제한, 질문 길이 제한
  - 진짜 비용 상한은 Claude Console의 지출 한도로 따로 설정
"""

import json
import os
import queue

os.environ.setdefault("DEMO_MODE", "1")  # 웹 데모는 기본이 데모 모드 (agent를 불러오기 전에 정해야 함)

import streamlit as st  # noqa: E402

from demo_runtime import AgentRuntime, Quota, ensure_index, extract_sources  # noqa: E402

SESSION_LIMIT = int(os.getenv("DEMO_SESSION_LIMIT", "5"))   # 방문자 한 명(세션)당 질문 수
DAILY_LIMIT = int(os.getenv("DEMO_DAILY_LIMIT", "60"))      # 하루 전체 질문 수
MAX_CHARS = 300
REPO_URL = os.getenv("REPO_URL", "https://github.com/EpenSia/devdocs-rag-agent")

EXAMPLES = [
    ("문서 검색", "응답을 보낸 뒤에 백그라운드 작업을 실행하려면 어떻게 하나요?"),
    ("문서 검색", "다른 도메인의 프론트엔드에서 내 API를 호출할 수 있게 하려면?"),
    ("문서 + GitHub 이슈", "BackgroundTasks 안에서 yield 의존성의 DB 세션이 이미 닫혀 있다는 오류가 나요. "
                         "원인과 관련 GitHub 이슈를 찾아 주세요."),
    ("이슈 초안 + 승인", "BackgroundTasks 문서에 'yield 의존성의 리소스를 백그라운드 작업에서 쓰면 안 된다'는 "
                        "주의 문구를 더 잘 보이게 해 달라는 문서 개선 요청을 sandbox 저장소에 이슈 초안으로 작성해 주세요."),
]

st.set_page_config(page_title="FastAPI Support Agent", page_icon="⚡", layout="wide")


# ── 앱 전체에서 한 번만 만드는 것들 ─────────────────────────
@st.cache_resource(show_spinner=False)
def get_runtime() -> AgentRuntime:
    ensure_index()
    return AgentRuntime()


@st.cache_resource(show_spinner=False)
def get_quota() -> Quota:
    return Quota(DAILY_LIMIT)


# ── 화면 조각 ─────────────────────────────────────────────
def short(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " …"


def render_event(ev: dict, box) -> None:
    """에이전트 진행 이벤트 하나를 화면 한 줄로."""
    kind = ev["type"]
    if kind == "thinking":
        box.caption(f"턴 {ev['turn']} · Claude가 다음 행동을 판단하는 중")
    elif kind == "think":
        box.markdown(f"💭 {short(ev['text'], 200)}")
    elif kind == "tool_use":
        args = short(json.dumps(ev["input"], ensure_ascii=False), 140)
        box.markdown(f"🔧 **{ev['server']}** · `{ev['name']}`  \n`{args}`")
    elif kind == "tool_result":
        box.caption(("❌ " if ev["is_error"] else "↳ ") + short(ev["text"]))
    elif kind == "draft":
        box.markdown("🛑 **이슈 초안 작성** → 사람 승인 대기 (아직 아무것도 실행하지 않음)")
    elif kind == "warn":
        box.warning(ev["text"])


def summary_label(m: dict) -> str:
    return (f"동작 과정 · 턴 {m['turns']} · 도구 {m['tools']}회 · "
            f"{m['tokens']:,} 토큰 · {m['seconds']}초")


def decide(msg_i: int, draft_i: int, decision: str) -> None:
    st.session_state.messages[msg_i]["drafts"][draft_i]["decision"] = decision


def render_draft(msg_i: int, draft_i: int, d: dict) -> None:
    with st.container(border=True):
        st.markdown(f"🛑 **승인 대기 · GitHub 이슈 초안** → `{d.get('owner')}/{d.get('repo')}`")
        st.markdown(f"**{d.get('title', '')}**")
        with st.expander("본문 보기", expanded=False):
            st.markdown(d.get("body", ""))
        decision = d.get("decision")
        if decision is None:
            c1, c2, _ = st.columns([1, 1, 4])
            c1.button("승인", key=f"ok-{msg_i}-{draft_i}", type="primary",
                      on_click=decide, args=(msg_i, draft_i, "approved"))
            c2.button("거절", key=f"no-{msg_i}-{draft_i}",
                      on_click=decide, args=(msg_i, draft_i, "rejected"))
            st.caption("공개 데모라 승인해도 실제로 등록되지 않습니다. 로컬 CLI에서는 승인한 뒤에만 실제 이슈가 생성됩니다.")
        elif decision == "approved":
            st.success("승인했습니다. 공개 데모라 GitHub에는 등록하지 않았습니다(드라이런).")
        else:
            st.warning("거절했습니다. 아무것도 실행되지 않았습니다.")


def render_assistant(msg_i: int, msg: dict) -> None:
    with st.expander(summary_label(msg["metrics"]), expanded=False):
        for ev in msg["events"]:
            render_event(ev, st)
    st.markdown(msg["content"])
    docs, issues = msg["sources"]
    if docs or issues:
        links = [f"[{label}]({url})" for label, url in docs[:6]]
        links += [f"[#{url.rsplit('/', 1)[-1]}]({url})" for url in issues[:5]]
        st.caption("근거 · " + " · ".join(links))
    for draft_i, d in enumerate(msg["drafts"]):
        render_draft(msg_i, draft_i, d)


def slim(ev: dict) -> dict:
    """세션에 보관할 이벤트는 도구 결과를 짧게 줄여 메모리를 아낀다."""
    if ev["type"] == "tool_result":
        return {**ev, "text": short(ev["text"], 300)}
    return ev


# ── 사이드바 ──────────────────────────────────────────────
def sidebar(quota: Quota) -> None:
    with st.sidebar:
        st.subheader("이 데모에서 볼 수 있는 것")
        st.markdown(
            "1. 에이전트가 **스스로 도구를 골라** 문서와 GitHub를 검색하는 과정\n"
            "2. 답변마다 붙는 **근거 링크** (FastAPI 공식 문서, GitHub 이슈)\n"
            "3. 이슈 초안의 **사람 승인 단계** (공개 데모라 실제 등록은 하지 않음)"
        )
        st.subheader("예시 질문")
        for i, (tag, q) in enumerate(EXAMPLES):
            if st.button(f"{tag} · {short(q, 46)}", key=f"ex-{i}", width="stretch"):
                st.session_state.pending = q
        st.subheader("사용량")
        used = st.session_state.asked
        st.progress(min(used / SESSION_LIMIT, 1.0), text=f"이 세션 {used}/{SESSION_LIMIT}개 사용")
        st.caption(f"오늘 전체 남은 질문: {quota.remaining()}개")
        st.subheader("평가 결과")
        st.markdown(
            "| 지표 | 값 |\n|---|---|\n"
            "| 검색 Hit@5 (100문항) | 96% |\n"
            "| 정답 문서 수신율 | 100% |\n"
            "| 근거 일치도 | 3.60 → 4.55 |"
        )
        st.caption("근거 일치도는 근거 규칙 적용 전후 20문항, 개선 16 · 악화 0 (부호 검정 p<0.001)")
        st.divider()
        st.markdown(f"[코드와 실험 기록 (GitHub)]({REPO_URL})")
        st.caption("질문은 Claude API로 전송됩니다. 개인정보는 입력하지 마세요.")


# ── 본문 ──────────────────────────────────────────────────
def main() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("asked", 0)
    quota = get_quota()

    st.title("⚡ FastAPI Support Agent")
    st.caption("FastAPI 공식 문서와 GitHub 이슈를 근거로 답하는 AI 에이전트 · "
               "RAG · MCP · Agent를 프레임워크 없이 직접 구현")

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY가 설정되지 않았습니다. 로컬은 .env, 배포 환경은 Secrets에 넣어 주세요.")
        st.stop()

    with st.spinner("에이전트를 준비하는 중입니다. 처음 실행이면 문서 인덱싱 때문에 2~3분 걸릴 수 있습니다…"):
        try:
            runtime = get_runtime()
        except Exception as e:
            st.error(f"에이전트를 시작하지 못했습니다: {e}")
            st.stop()

    sidebar(quota)

    if not st.session_state.messages:
        st.info("왼쪽의 **예시 질문**을 누르거나 아래에 직접 질문해 보세요. 한국어로 물어도 영어 문서를 찾아 한국어로 답합니다. "
                "각 답변 위의 **동작 과정**을 펼치면 에이전트가 어떤 도구를 어떤 순서로 불렀는지 볼 수 있습니다.")
        st.caption(f"연결된 도구: {', '.join(f'`{n}`' for n in runtime.tool_names)}")

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant(i, msg)

    typed = st.chat_input(f"FastAPI에 대해 질문해 보세요 (최대 {MAX_CHARS}자)", max_chars=MAX_CHARS)
    question = (typed or st.session_state.pop("pending", None) or "").strip()
    if not question:
        return

    # 공개 데모의 비용 보호: 세션 제한 → 하루 전체 제한 순서로 확인
    if st.session_state.asked >= SESSION_LIMIT:
        st.warning(f"이 세션에서는 질문을 {SESSION_LIMIT}개까지 할 수 있습니다. 코드는 GitHub에서 확인해 주세요.")
        return
    if not quota.take():
        st.warning("오늘 데모 사용량을 모두 썼습니다. 내일 다시 시도해 주세요.")
        return
    st.session_state.asked += 1

    st.session_state.messages.append({"role": "user", "content": question[:MAX_CHARS]})
    with st.chat_message("user"):
        st.markdown(question[:MAX_CHARS])

    with st.chat_message("assistant"):
        status = st.status("에이전트가 작업 중입니다…", expanded=True)
        events_q: queue.Queue = queue.Queue()
        future = runtime.ask(question[:MAX_CHARS], events_q.put)
        events = []
        while True:  # 백그라운드에서 오는 진행 이벤트를 받아 바로 화면에 그린다
            try:
                ev = events_q.get(timeout=0.25)
            except queue.Empty:
                if future.done():
                    break
                continue
            events.append(slim(ev))
            render_event(ev, status)
        try:
            answer, trace = future.result()
        except Exception as e:
            status.update(label="오류가 발생했습니다", state="error", expanded=True)
            st.error(f"에이전트 실행 중 오류: {short(e, 300)}")
            return

    metrics = {"turns": trace["turns"], "tools": len(trace["tool_calls"]),
               "tokens": trace["input_tokens"] + trace["output_tokens"], "seconds": trace.get("seconds", 0)}
    st.session_state.messages.append({
        "role": "assistant", "content": answer, "events": events, "metrics": metrics,
        "sources": extract_sources(trace["tool_results"]),
        "drafts": [{**d, "decision": None} for d in trace.get("drafts", [])],
    })
    st.rerun()  # 기록에서 다시 그려 승인 버튼 등을 한 곳에서 처리


main()
