"""
5단계: 검색 품질 평가 (LLM 호출 없음 → 비용 0원)
------------------------------------------------
eval_set.json의 질문마다 '정답 문서'가 검색 결과 몇 위에 나오는지 측정한다.

지표:
  Hit@1  : 1위 결과가 정답 문서인 비율
  Hit@5  : 상위 5개 안에 정답 문서가 있는 비율
  MRR    : 정답 순위의 역수 평균 (1위=1.0, 2위=0.5, 5위=0.2, 없음=0) → 높을수록 정답이 위에 있음
  Noise  : 상위 5개 중 release-notes.md 청크 비율 → 낮을수록 좋음

사용법:
  uv run eval_retrieval.py <라벨>             # 측정 → eval_log.md에 표 기록, eval_runs/<라벨>.json에 질문별 순위 저장
  uv run eval_retrieval.py compare <A> <B>    # 두 실행에서 순위가 바뀐 질문만 보여줌
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from rag_basic import BASE_DIR, retrieve

MODES = ["vector", "bm25", "hybrid"]
K = 5
RUNS_DIR = BASE_DIR / "eval_runs"


def rank_of(hits, expected: list[str]) -> int | None:
    """정답 문서가 처음 나온 순위(1부터). 없으면 None."""
    for r, h in enumerate(hits, 1):
        src = h["source"].replace("\\", "/")
        if any(src.endswith(e) for e in expected):
            return r
    return None


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
        "MRR": sum(1 / r for r in ranks if r) / n,
        "Noise": sum(p["noise"] for p in per_q) / n,
    }
    return metrics, per_q


def run(label: str):
    cases = json.loads((BASE_DIR / "eval_set.json").read_text(encoding="utf-8"))
    results, rows = {}, []
    for mode in MODES:
        metrics, per_q = evaluate(cases, mode)
        results[mode] = {"metrics": metrics, "questions": per_q}
        rows.append(f"| {mode} | {metrics['Hit@1']:.0%} | {metrics['Hit@5']:.0%} | "
                    f"{metrics['MRR']:.3f} | {metrics['Noise']:.0%} |")

    table = "\n".join([f"| {label} ({len(cases)}문항) | Hit@1 | Hit@5 | MRR | Noise |",
                       "|---|---|---|---|---|", *rows])
    print("\n" + table)

    print("\n── hybrid에서 상위 5개 안에 정답이 없었던 질문 ──")
    misses = [p for p in results["hybrid"]["questions"] if p["rank"] is None]
    for p in misses or [{"q": "(없음)", "top3": []}]:
        print(f"- {p['q']}\n    상위 결과: {p['top3']}")

    RUNS_DIR.mkdir(exist_ok=True)
    (RUNS_DIR / f"{label}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    with open(BASE_DIR / "eval_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### {label} — {datetime.now():%Y-%m-%d %H:%M}\n\n{table}\n")
    print(f"\n📝 eval_log.md에 기록, 질문별 결과는 eval_runs/{label}.json에 저장했습니다.")


def compare(a: str, b: str):
    """같은 질문의 순위가 A → B에서 어떻게 바뀌었는지. 평균에 가려진 원인을 찾기 위함."""
    ra = json.loads((RUNS_DIR / f"{a}.json").read_text(encoding="utf-8"))
    rb = json.loads((RUNS_DIR / f"{b}.json").read_text(encoding="utf-8"))
    fmt = lambda r: f"{r}위" if r else "없음"
    for mode in MODES:
        print(f"\n── {mode}: {a} → {b} ──")
        changed = 0
        for qa, qb in zip(ra[mode]["questions"], rb[mode]["questions"]):
            if qa["rank"] == qb["rank"]:
                continue
            changed += 1
            better = (qb["rank"] or 99) < (qa["rank"] or 99)
            print(f"{'▲' if better else '▼'} {fmt(qa['rank']):>3} → {fmt(qb['rank']):<3} {qa['q']}")
            if not better:
                print(f"      {b} 상위 결과: {qb['top3']}")
        print(f"(순위가 바뀐 질문 {changed}개)" if changed else "(변화 없음)")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1] if len(sys.argv) > 1 else "run")
