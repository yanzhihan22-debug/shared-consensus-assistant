"""库内照片自测 — 留一法：每张图当查询，排除自身，看匹配是否仍命中同型号"""
import os
import numpy as np
import matcher

def leave_one_out(index, photos_map):
    """对每张照片查询，排除该图自身（若同型号只有1张图则跳过）"""
    results = []
    for key, photos in photos_map.items():
        for i, p in enumerate(photos):
            fp = os.path.join(matcher.BASE_DIR, p.replace("\\", os.sep))
            if not os.path.exists(fp):
                continue
            # 排除自身做查询
            q = matcher._encode(fp)
            scores = []
            for k2, vecs in index.items():
                sims = vecs @ q
                # 排除同一张照片的匹配
                if k2 == key and i < len(vecs):
                    sims = np.delete(sims, i)
                if sims.size > 0:
                    scores.append((k2, float(np.max(sims))))
                else:
                    scores.append((k2, float('-inf')))
            scores.sort(key=lambda x: x[1], reverse=True)
            top = scores[0]
            ok = top[0] == key
            results.append((key, p, top[0], top[1], ok))
    return results

def main():
    db = matcher.load_manual_db()
    photos_map = {k: e.get("photos", []) for k, e in db.items()}
    results = leave_one_out(matcher.CLIP_INDEX, photos_map)
    total = len(results)
    correct = sum(1 for r in results if r[4])
    print("\n========== CLIP 留一法自测 ==========")
    print(f"测试数: {total}  正确: {correct}  准确率: {correct/max(total,1)*100:.0f}%\n")
    for name, p, pred, score, ok in results:
        mark = "OK" if ok else "X"
        print(f"{mark} {name:14s} {os.path.basename(p):20s} -> {pred:14s} ({score:.3f})")
    print("======================================")

if __name__ == "__main__":
    main()
