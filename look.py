import json, re
rows = [json.loads(l) for l in open("behavior/samples.jsonl")]
for r in rows:
    if r["s"] not in (0.0, 0.5):          # 先只看 s=0 和 s=0.5
        continue
    t = r["text"]
    w = re.findall(r"\w+", t.lower())
    half = w[len(w) // 2:]
    g = [tuple(half[i:i + 4]) for i in range(len(half) - 3)]
    rep = 1 - len(set(g)) / max(1, len(g))
    print("=" * 80)
    print(f"s={r['s']}  topic: {r['topic']}  后半段词级重复率={rep:.3f}")
    print("--- 开头 ---"); print(t[:500])
    print("--- 结尾 ---"); print(t[-800:])