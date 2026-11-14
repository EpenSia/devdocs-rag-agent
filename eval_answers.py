"""
6단계: 답변 품질 평가 (에이전트 실행 + LLM 채점)
------------------------------------------------
검색 평가(eval_retrieval.py)는 "정답 문서를 찾았나"만 본다.
여기서는 에이전트가 실제로 낸 '최종 답변'을 평가한다.

  자동 계산 (LLM 불필요)
    Retrieved : 에이전트가 도구로 받은 결과 안에 정답 문서가 들어 있었나
    Cited     : 최종 답변이 정답 문서를 출처로 언급했나
    Turns / Tools / Tokens / Seconds : 몇 번 돌았고, 도구를 몇 번 불렀고, 비용과 시간이 얼마였나

  LLM 채점 (Judge, 1~5점)
    Faithfulness : 답변의 주장과 코드가 '에이전트가 받은 도구 결과'로 뒷받침되는가 (근거 없는 내용 = 감점)
    Relevance    : 질문에 직접적이고 충분하게 답했는가

    Korean       : 한국어로 답했나 (LLM 채점자는 언어를 보지 않아 자동 검사로 따로 확인)
    Truncated    : 답변이 최대 길이에서 잘린 개수
    Labeled      : 일반 지식을 "문서 외 참고" 섹션에 따로 표시한 답변 비율 (그 섹션 안의 내용은 감점하지 않음)
    ReadDoc      : 답변 전에 원문(read_doc)을 확인한 평균 횟수

사용법:
  uv run eval_answers.py <라벨> [--n=20]
  → eval_log.md에 요약 기록, eval_answers/<라벨>.json에 질문별 상세 저장
  uv run eval_answers.py compare <A> <B>   # 같은 질문끼리 Faithfulness 비교 + 부호 검정 (API 호출 없음)

비용: 질문 1개당 에이전트 호출 약 2~4회 + 채점 1회. --n으로 질문 수를 조절하세요.
한계: 채점자도 LLM이라 완벽하지 않음 → 점수가 낮은 답변은 사람이 직접 읽어 확인.
"""

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path

os.environ["GITHUB_TOKEN"] = ""  # 문서 검색만으로 평가 (GitHub 결과는 매번 달라져 재현이 어려움)
os.environ["AGENT_QUIET"] = "1"

import agent  # noqa: E402  (위 환경변수를 먼저 설정해야 함)
from anthropic import Anthropic  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "eval_answers"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", agent.MODEL)
MAX_CONTEXT_CHARS = 40_000

JUDGE_PROMPT = """You grade an answer written by a FastAPI documentation assistant.
The assistant could only use the CONTEXT below (what its search tools returned).

The answer may end with a section titled "{outside_title}" (= "outside the docs").
Claims inside that section are explicitly labeled as general knowledge: do NOT count them as unsupported.

Score each from 1 to 5:
- faithfulness: Are the factual claims and code OUTSIDE that labeled section supported by the CONTEXT?
  5 = everything supported; 3 = some claims or code not in the context; 1 = mostly unsupported.
  Unlabeled general knowledge not in the context counts as unsupported.
- relevance: Does the answer directly and sufficiently address the QUESTION? 5 = fully, 1 = not at all.
List up to 3 short unsupported claims (empty list if none).

Reply with JSON only: {{"faithfulness": <int>, "relevance": <int>, "unsupported": [<str>, ...]}}

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}"""


def parse_json(text: str) -> dict:
    """응답 속 첫 번째 '올바른' JSON 객체를 찾는다.
    (6단계 버그: 정규식 r'\\{.*\\}'는 첫 '{'부터 마지막 '}'까지 잡아서, 채점 이유에 중괄호가 섞이면 실패했음)"""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "faithfulness" in obj:
            return obj
    return {}


def judge(llm, question: str, context: str, answer: str, tries: int = 2) -> dict:
    # 참고: Anthropic Python SDK 1.0(2026-08)부터 temperature 파라미터가 없음 → 채점이 실행마다 약간 달라질 수 있음
    prompt = JUDGE_PROMPT.format(outside_title=agent.OUTSIDE_DOCS_TITLE, question=question,
                                 context=context[:MAX_CONTEXT_CHARS], answer=answer)
    text = ""
    for _ in range(tries):  # 형식이 깨지면 한 번 더 시도
        resp = llm.messages.create(model=JUDGE_MODEL, max_tokens=1000,
                                   messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        score = parse_json(text)
        if score:
            return score
    return {"judge_raw": text[:2000]}  # 실패 원인을 나중에 볼 수 있도록 원문 보관


def language_ok(answer: str) -> bool:
    """한국어 답변인가. 에이전트의 언어 가드레일과 같은 기준(agent.is_korean)을 사용.
    LLM 채점자는 언어를 일관되게 보지 않아 규칙으로 따로 검사."""
    return agent.is_korean(answer)


def pick(cases: list[dict], n: int) -> list[dict]:
    """전체 질문에서 고르게 n개 선택 (앞쪽에 몰리지 않게, 매번 같은 질문)."""
    step = max(len(cases) / n, 1)
    return [cases[int(i * step)] for i in range(min(n, len(cases)))]


def mentions(text: str, expected: list[str]) -> bool:
    """정답 문서가 언급됐는지. 'tutorial/cors.md' 또는 'cors.md'로 인용해도 인정 (index.md는 경로까지 요구)."""
    norm = text.replace("\\", "/")
    for e in expected:
        keys = [e]  # 확장자까지 비교 ('async'만 비교하면 'async def'도 걸림)
        if Path(e).stem != "index":
            keys.append(Path(e).name)
        if any(k in norm for k in keys):
            return True
    return False


async def main(label: str, n: int):
    cases = pick(json.loads((BASE_DIR / "eval_set.json").read_text(encoding="utf-8")), n)
    llm = Anthropic()
    rows = []
    async with AsyncExitStack() as stack:
        registry, tools = await agent.connect_servers(stack)
        for i, c in enumerate(cases, 1):
            trace: dict = {}
            try:
                answer = await agent.run_agent(c["q"], registry, tools, llm, trace)
            except Exception as e:
                answer = f"(error: {e})"
            context = "\n\n---\n\n".join(trace.get("tool_results", []))
            score = judge(llm, c["q"], context, answer)
            row = {
                "q": c["q"], "expected": c["expected"], "answer": answer,
                "retrieved": mentions(context, c["expected"]), "cited": mentions(answer, c["expected"]),
                "faithfulness": score.get("faithfulness"), "relevance": score.get("relevance"),
                "unsupported": score.get("unsupported", []),
                "judge_raw": score.get("judge_raw"),  # 채점 실패 시에만 값이 있음
                "korean": language_ok(answer),
                "language_fixed": trace.get("language_fixed", False),  # 가드레일이 다시 쓰게 했나
                "truncated": trace.get("truncated", False),
                "labeled": agent.OUTSIDE_DOCS_TITLE in answer,  # 일반 지식을 따로 표시했나
                "turns": trace.get("turns", 0), "tool_calls": len(trace.get("tool_calls", [])),
                "tools": [t["name"] for t in trace.get("tool_calls", [])],
                "read_doc": [t["name"] for t in trace.get("tool_calls", [])].count("read_doc"),
                "context": context,  # 나중에 채점 기준만 바꿔 다시 채점할 수 있도록 저장
                "tokens": trace.get("input_tokens", 0) + trace.get("output_tokens", 0),
                "seconds": trace.get("seconds", 0),
            }
            rows.append(row)
            flags = ("" if row["korean"] else " ⚠️한국어아님") + (" 🔁언어재작성" if row["language_fixed"] else "") \
                + (" ⚠️잘림" if row["truncated"] else "") \
                + (" ⚠️채점실패" if row["faithfulness"] is None else "")
            print(f"[{i}/{len(cases)}] F{row['faithfulness']} R{row['relevance']} "
                  f"{'✓' if row['retrieved'] else '✗'}검색 {'✓' if row['cited'] else '✗'}인용 "
                  f"turns={row['turns']} tools={row['tools']}{flags}  {c['q'][:60]}")

    # 채점 실패(None)는 0점으로 세지 않고 평균에서 제외 (6단계 버그: 0점으로 세어 평균이 낮게 나왔음)
    judged = [r for r in rows if r["faithfulness"] is not None]
    avg = lambda k, rs=rows: sum(r[k] or 0 for r in rs) / max(len(rs), 1)
    rate = lambda k: sum(bool(r[k]) for r in rows) / len(rows)
    table = "\n".join([
        f"| {label} ({len(rows)}문항) | Faithfulness | Relevance | Retrieved | Cited | Korean | Truncated "
        "| Labeled | ReadDoc | Turns | Tools | Tokens | Seconds |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| 평균 | {avg('faithfulness', judged):.2f} | {avg('relevance', judged):.2f} | {rate('retrieved'):.0%} | "
        f"{rate('cited'):.0%} | {rate('korean'):.0%} | {sum(r['truncated'] for r in rows)}개 | "
        f"{rate('labeled'):.0%} | {avg('read_doc'):.1f} | {avg('turns'):.1f} | "
        f"{avg('tool_calls'):.1f} | {avg('tokens'):,.0f} | {avg('seconds'):.1f} |",
    ])
    if len(judged) < len(rows):
        table += f"\n\n채점 실패 {len(rows) - len(judged)}개는 Faithfulness·Relevance 평균에서 제외"
    if any(r["language_fixed"] for r in rows):
        table += f"\n\n언어 가드레일이 다시 쓰게 한 답변: {sum(r['language_fixed'] for r in rows)}개"
    print("\n" + table)

    weak = [r for r in rows if r["faithfulness"] is None or r["faithfulness"] <= 3 or (r["relevance"] or 0) <= 3
            or not r["korean"] or r["truncated"]]
    print(f"\n── 사람이 직접 읽어 확인할 답변 {len(weak)}개 (3점 이하 · 한국어 아님 · 잘림 · 채점 실패) ──")
    for r in weak:
        why = [w for w, bad in (("한국어 아님", not r["korean"]), ("잘림", r["truncated"]),
                                ("채점 실패", r["faithfulness"] is None)) if bad]
        print(f"- F{r['faithfulness']} R{r['relevance']} {r['q']}" + (f"  [{', '.join(why)}]" if why else ""))
        if r["unsupported"]:
            print(f"    근거 없는 내용: {r['unsupported']}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{label}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(BASE_DIR / "eval_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### 답변 평가: {label} — {datetime.now():%Y-%m-%d %H:%M} "
                f"(judge: {JUDGE_MODEL}, search: {os.getenv('SEARCH_MODE', 'hybrid')})\n\n{table}\n")
    print(f"\n📝 eval_log.md에 기록, 상세는 eval_answers/{label}.json에 저장했습니다.")


def compare(a: str, b: str):
    """두 답변 평가 실행을 같은 질문끼리 비교 + 부호 검정. 채점에 temperature가 없어 평균만으로 판단하지 않기 위함."""
    from eval_retrieval import sign_test  # 검색 평가와 같은 검정 사용

    ra = json.loads((OUT_DIR / f"{a}.json").read_text(encoding="utf-8"))
    rb = {r["q"]: r for r in json.loads((OUT_DIR / f"{b}.json").read_text(encoding="utf-8"))}
    common = [(x, rb[x["q"]]) for x in ra if x["q"] in rb]
    # 어느 한쪽이라도 채점에 실패한 질문은 비교에서 제외 (0점으로 세면 가짜 '악화'가 생김)
    pairs = [(x, y) for x, y in common if x["faithfulness"] is not None and y["faithfulness"] is not None]
    skipped = len(common) - len(pairs)
    print(f"── {a} → {b} (공통 질문 {len(common)}개" + (f", 채점 실패 {skipped}개 제외" if skipped else "") + ") ──")
    up = down = 0
    for x, y in pairs:
        fx, fy = x["faithfulness"], y["faithfulness"]
        if fx == fy:
            continue
        up, down = up + (fy > fx), down + (fy < fx)
        extra = f"  read_doc {y.get('read_doc', 0)}회" if y.get("read_doc") else ""
        print(f"{'▲' if fy > fx else '▼'} F{fx} → F{fy}  {x['q'][:70]}{extra}")
        if fy < fx:
            print(f"      {b}의 근거 없는 내용: {y['unsupported']}")
    p = sign_test(up, down)
    mean = lambda rows, k: sum((r.get(k) or 0) for r in rows) / max(len(rows), 1)
    xs, ys = [x for x, _ in pairs], [y for _, y in pairs]
    print(f"\nFaithfulness: 개선 {up} · 악화 {down} · 같음 {len(pairs) - up - down} → p={p:.3f} "
          f"({'우연이라고 보기 어려움' if p < 0.05 else '차이가 있다고 말할 근거 부족'})")
    for k in ("faithfulness", "relevance", "read_doc", "tokens", "seconds"):
        print(f"  {k:<12} {mean(xs, k):>9.2f} → {mean(ys, k):>9.2f}")
    # 언어와 잘림은 예전 실행 파일에 없을 수 있으므로 답변 원문에서 다시 계산
    not_ko = lambda rows: sum(not language_ok(r["answer"]) for r in rows)
    cut = lambda rows: sum(bool(r.get("truncated")) or "길이 제한으로 중간에 잘렸" in r["answer"] for r in rows)
    allx, ally = [x for x, _ in common], [y for _, y in common]
    print(f"  {'한국어 아님':<10} {not_ko(allx):>9} → {not_ko(ally):>9}")
    print(f"  {'잘린 답변':<11} {cut(allx):>9} → {cut(ally):>9}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
        sys.exit()
    args = sys.argv[1:]
    n_arg = next((a.split("=", 1)[1] for a in args if a.startswith("--n=")), "20")
    if not n_arg.isdigit():
        sys.exit(f"--n 값은 숫자여야 합니다: '{n_arg}' (명령어가 두 번 붙여넣어지지 않았는지 확인하세요)")
    n = int(n_arg)
    names = [a for a in args if not a.startswith("--")]
    asyncio.run(main(names[0] if names else "answers", n))
