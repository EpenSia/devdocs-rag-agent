"""
5·6단계: 검색 품질 평가 (LLM 호출 없음 → 비용 0원)
------------------------------------------------
eval_set.json의 질문마다 '정답 문서'가 검색 결과 몇 위에 나오는지 측정한다.

지표:
  Hit@1  : 1위 결과가 정답 문서인 비율
  Hit@5  : 상위 5개 안에 정답 문서가 있는 비율
  MRR    : 정답 순위의 역수 평균 (1위=1.0, 2위=0.5, 5위=0.2, 없음=0) → 높을수록 정답이 위에 있음
  Noise  : 상위 5개 중 release-notes.md 청크 비율 → 낮을수록 좋음

6단계 추가: 짝지은 비교 + 부호 검정(sign test)
  같은 질문에서 A가 이긴 수 / B가 이긴 수를 세고, 그 차이가 우연으로 나올 확률(p값)을 계산한다.
  p < 0.05 이면 "우연이라고 보기 어렵다", 그 이상이면 "차이가 있다고 말할 근거가 부족하다".

사용법:
  uv run eval_retrieval.py <라벨>             # 측정 → eval_log.md 기록, eval_runs/<라벨>.json 저장
  uv run eval_retrieval.py compare <A> <B>    # 두 실행 비교: 순위가 바뀐 질문 + 부호 검정
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

from rag_basic import BASE_DIR, retrieve

MODES = ["vector", "bm25", "hybrid"]
K = 5
RUNS_DIR = BASE_DIR / "eval_runs"
DOCS_DIR = BASE_DIR / "data" / "fastapi_docs"


def rank_of(hits, expected: list[str]) -> int | None:
    """정답 문서가 처음 나온 순위(1부터). 없으면 None."""
    for r, h in enumerate(hits, 1):
        src = h["source"].replace("\\", "/")
        if any(src.endswith(e) for e in expected):
            return r
    return None


def rr(rank) -> float:
    return 1 / rank if rank else 0.0


def sign_test(wins: int, losses: int) -> float:
    """양측 부호 검정 p값: 승패가 반반(동전 던지기)이라고 가정했을 때 이만큼 치우칠 확률."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n * 2
    return min(p, 1.0)


def paired(qa: list[dict], qb: list[dict]) -> tuple[int, int, int, float]:
    """같은 질문끼리 순위(역수)를 비교. 반환: (B가 나은 수, A가 나은 수, 같음, p값)"""
    b_better = sum(rr(b["rank"]) > rr(a["rank"]) for a, b in zip(qa, qb))
    a_better = sum(rr(a["rank"]) > rr(b["rank"]) for a, b in zip(qa, qb))
    return b_better, a_better, len(qa) - b_better - a_better, sign_test(b_better, a_better)


def verdict(p: float) -> str:
    return "우연이라고 보기 어려움 (p<0.05)" if p < 0.05 else "차이가 있다고 말할 근거 부족"


def load_cases():
    cases = json.loads((BASE_DIR / "eval_set.json").read_text(encoding="utf-8"))
    ok = [c for c in cases if any((DOCS_DIR / e).is_file() for e in c["expected"])]
    if len(ok) < len(cases):  # 내 문서 사본에 없는 페이지를 묻는 질문은 공정하지 않으므로 제외
        print(f"⚠️  정답 문서가 data/fastapi_docs에 없는 질문 {len(cases) - len(ok)}개 제외")
    return ok


def evaluate(cases, mode):
    """반환: (지표 dict, 질문별 결과 list)"""
    per_q = []
    for c in cases:
        hits = retrieve(c["q"], k=K, mode=mode)
        per_q.append({
            "q": c["q"],
            "rank": rank_of(hits, c["expected"]),
            "top3": [Path(h["source"].replace("\\", "/")).name for h in hits[:3]],
            "noise": sum("release-notes" in h["source"] for h in hits) / K,
        })
    n = len(per_q)
    ranks = [p["rank"] for p in per_q]
    metrics = {
        "Hit@1": sum(r == 1 for r in ranks) / n,
        "Hit@5": sum(r is not None for r in ranks) / n,
        "MRR": sum(rr(r) for r in ranks) / n,
        "Noise": sum(p["noise"] for p in per_q) / n,
    }
    return metrics, per_q


def run(label: str):
    cases = load_cases()
    results, rows = {}, []
    for mode in MODES:
        metrics, per_q = evaluate(cases, mode)
        results[mode] = {"metrics": metrics, "questions": per_q}
        rows.append(f"| {mode} | {metrics['Hit@1']:.0%} | {metrics['Hit@5']:.0%} | "
                    f"{metrics['MRR']:.3f} | {metrics['Noise']:.0%} |")

    table = "\n".join([f"| {label} ({len(cases)}문항) | Hit@1 | Hit@5 | MRR | Noise |",
                       "|---|---|---|---|---|", *rows])
    w, l, t, p = paired(results["vector"]["questions"], results["hybrid"]["questions"])
    test = (f"hybrid vs vector (질문별 순위 비교): hybrid 우세 {w} · vector 우세 {l} · 같음 {t} "
            f"→ p={p:.3f}, {verdict(p)}")
    print("\n" + table + "\n\n" + test)

    print("\n── hybrid에서 상위 5개 안에 정답이 없었던 질문 ──")
    misses = [q for q in results["hybrid"]["questions"] if q["rank"] is None]
    for q in misses or [{"q": "(없음)", "top3": []}]:
        print(f"- {q['q']}\n    상위 결과: {q['top3']}")

    RUNS_DIR.mkdir(exist_ok=True)
    (RUNS_DIR / f"{label}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    with open(BASE_DIR / "eval_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### {label} — {datetime.now():%Y-%m-%d %H:%M}\n\n{table}\n\n{test}\n")
    print(f"\n📝 eval_log.md에 기록, 질문별 결과는 eval_runs/{label}.json에 저장했습니다.")


def compare(a: str, b: str):
    """같은 질문의 순위가 A → B에서 어떻게 바뀌었는지 + 부호 검정. (질문 문장으로 짝지음)"""
    ra = json.loads((RUNS_DIR / f"{a}.json").read_text(encoding="utf-8"))
    rb = json.loads((RUNS_DIR / f"{b}.json").read_text(encoding="utf-8"))
    fmt = lambda r: f"{r}위" if r else "없음"
    for mode in MODES:
        qb_by_text = {q["q"]: q for q in rb[mode]["questions"]}
        pairs = [(qa, qb_by_text[qa["q"]]) for qa in ra[mode]["questions"] if qa["q"] in qb_by_text]
        print(f"\n── {mode}: {a} → {b} (공통 질문 {len(pairs)}개) ──")
        for qa, qb in pairs:
            if qa["rank"] == qb["rank"]:
                continue
            better = rr(qb["rank"]) > rr(qa["rank"])
            print(f"{'▲' if better else '▼'} {fmt(qa['rank']):>3} → {fmt(qb['rank']):<3} {qa['q']}")
            if not better:
                print(f"      {b} 상위 결과: {qb['top3']}")
        w, l, t, p = paired([x for x, _ in pairs], [y for _, y in pairs])
        print(f"→ 개선 {w} · 악화 {l} · 같음 {t}, p={p:.3f}: {verdict(p)}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1] if len(sys.argv) > 1 else "run")
