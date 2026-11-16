"""纯淘宝识图批量自测：对 MANUAL_DB 全部照片逐一识图，统计准确率"""
import os
import sys
import json
import time
import matcher
import taobao_search
from playwright.sync_api import sync_playwright

def norm(s):
    import re
    return re.sub(r"[\s\-_]", "", str(s).upper())

def main():
    db = matcher.load_manual_db()
    photos_map = {k: e.get("photos", []) for k, e in db.items()}
    report = []
    correct = total = 0
    with sync_playwright() as p:
        browser, context = taobao_search._new_context(p)
        try:
            for key in sorted(photos_map):
                lib_name = db[key].get('name', key)
                photos = photos_map[key]
                for rel in photos:
                    fp = os.path.join(matcher.BASE_DIR, rel.replace("\\", os.sep))
                    if not os.path.exists(fp):
                        continue
                    total += 1
                    print(f"[{total}] {key} / {os.path.basename(fp)} ...", flush=True)
                    try:
                        r = taobao_search.search_image(fp, browser=browser, context=context)
                        pred = r.get("name")
                        # 模糊判定：归一化后子串互相包含（ESP32-S3 对 esp32-s3-cam、MAX98357 对 MAX98357A）
                        ok = False
                        if pred:
                            pn, ln = norm(pred), norm(lib_name)
                            ok = pn == ln or pn in ln or ln in pn
                        if ok:
                            correct += 1
                        report.append((key, os.path.basename(fp), pred, r.get("votes"), r.get("total"), ok, r.get("error")))
                        print(f"    当选={pred} (votes={r.get('votes')}/{r.get('total')}) {'OK' if ok else 'X'}", flush=True)
                    except Exception as e:
                        report.append((key, os.path.basename(fp), None, 0, 0, False, str(e)))
                        print(f"    异常: {e}", flush=True)
                    time.sleep(1)
        finally:
            browser.close()

    print("\n========== 淘宝识图批量自测 ==========")
    print(f"测试数: {total}  正确: {correct}  准确率: {correct/max(total,1)*100:.0f}%\n")
    for key, name, pred, votes, ttl, ok, err in report:
        mark = "OK" if ok else "X"
        err_s = f" [err:{err}]" if err else ""
        print(f"{mark} {key:14s} {name:20s} -> {str(pred):14s} (votes={votes}/{ttl}){err_s}")
    with open("taobao_batch_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n报告已保存 taobao_batch_report.json")

if __name__ == "__main__":
    main()
