"""淘宝以图搜图（拍立淘）识图模块

原理：Playwright 驱动真实 Chromium（有头模式，抗反爬），向 s.taobao.com
搜索页的隐藏 file input 注入图片（等价拟人"选择文件上传"），等结果加载后
抓取所有商品标题，聚合统计库内元件名出现次数，票数最高者当选。

用法（命令行）：
    python taobao_search.py <图片路径>
    python taobao_search.py <图片路径> --candidates "ST7789,SG90,ESP32-S3-CAM"

导入：
    from taobao_search import search_image
    result = search_image("a.jpg")   # -> {"name": ..., "votes": n, "total": m}
"""
import os
import re
import sys
import json
import argparse
from collections import Counter

from playwright.sync_api import sync_playwright

STATE_FILE = os.path.join(os.path.dirname(__file__), "taobao_state.json")
SEARCH_URL = "https://s.taobao.com/search"
DEFAULT_CANDIDATES = [
    "ST7789", "SG90", "ESP32-S3-CAM", "INMP441",
    "MAX98357A", "BREAD BOARD", "BREADBOARD", "面包板",
    "ST7735", "TFT", "LCD", "OLED", "舵机", "伺服", "麦克风", "功放",
    "LED灯珠", "LED灯", "LED", "灯珠", "发光二极管",
]

# 中文 → 英文自动映射（LED 只支持 ASCII，识别出中文名时自动转英文）
_CN_TO_EN = {
    "面包板": "BREADBOARD",
    "麦克风": "MIC",
    "舵机": "SERVO",
    "伺服": "SERVO",
    "功放": "AMP",
    "开发板": "DEVBOARD",
    "模块": "MODULE",
    "传感器": "SENSOR",
    "LED灯珠": "LED",
    "LED灯": "LED",
    "灯珠": "LED",
    "发光二极管": "LED",
}

# 归一化：去掉空格/连字符/下划线，统一大写，便于子串匹配
def norm(s):
    return re.sub(r"[\s\-_]", "", s).upper()

# 候选归一化后的长度，匹配时短候选优先（避免 "TFT" 匹配一切）
_CAND_NORM = sorted(
    [(norm(c), c) for c in DEFAULT_CANDIDATES],
    key=lambda x: -len(x[0]),
)

# 型号模式：字母开头 + 数字（可含变体 A/V/E 后缀），如 MAX98357A / ST7789 / INMP441 / ESP32-S3
# 排除常见噪声词（颜色/尺寸/连接词）
_MODEL_RE = re.compile(r"\b[A-Z]{2,}[0-9][A-Z0-9\-]*[A-Z0-9]?\b")
_NOISE = {
    "TFT", "LCD", "OLED", "IPS", "SPI", "I2C", "IIC", "USB", "DIY",
    "AI", "ARM", "MCU", "PCB", "GPS", "RGB", "WIFI", "BLE", "MEMS", "HMI",
    "LVGL", "PWM", "ADC", "DAC", "SDA", "SCL", "TQFN", "BGA", "VCC", "GND",
    "3W", "5V", "3V3", "0V", "PIE", "T0", "T1", "N", "S", "A", "M", "B",
}


def _extract_models(titles):
    """从原始标题（保留空格）提取高频型号词。返回 [(model, count)] 按票数降序。

    在未 norm 的标题上匹配，避免空格拼接产生假型号；
    每次取标题里最长的型号串（更完整，如 MAX98357AETE 优先于 MAX98357）。
    """
    counter = Counter()
    for t in titles:
        best = None
        for m in _MODEL_RE.findall(t):
            m2 = m.strip("-").strip()
            if not m2 or m2 in _NOISE or len(m2) < 4:
                continue
            if best is None or len(m2) > len(best):
                best = m2
        if best:
            counter[best] += 1
    return counter.most_common()


def _map_to_library(model):
    return model  # 已不依赖本地库，原样返回


def _cn_to_en(name):
    """中文型号自动转英文（LED 只支持 ASCII）"""
    if not name:
        return name
    for cn, en in _CN_TO_EN.items():
        if cn in name:
            print(f"   [中文映射] {name} -> {en}", flush=True)
            return en
    # 全中文但没有映射的，尝试保留可能存在的英文部分
    if re.search(r'[\u4e00-\u9fff]', name) and not re.search(r'[A-Za-z0-9]', name):
        return "COMPONENT"  # 全中文无映射时的兜底 ASCII 名
    return name


def _wait_titles(page, timeout_ms=30000):
    """上传后：等上传完成 → 点"搜索"键 → 等待并返回结果标题列表（去重）。"""
    # 1) 等待"正在上传图片"消失（上传完成），给点搜索前留时间
    try:
        page.wait_for_function(
            """() => {
                const t = document.body.innerText || '';
                return !/正在上传图片/.test(t);
            }""",
            timeout=25000,
        )
    except Exception:
        pass
    page.wait_for_timeout(1500)
    # 2) 点击"搜索"按钮（上传完成后的搜索键）
    for sel in ["div.upload-button.upload-button-active", "button.btn-search", "button.searchBtn--lcduFNeM"]:
        try:
            btn = page.query_selector(sel)
            if btn and btn.is_visible():
                btn.click()
                print("   [已点击搜索按钮]", flush=True)
                break
        except Exception:
            continue
    page.wait_for_timeout(6000)  # 结果渲染
    try:
        page.wait_for_function(
            """() => {
                const links = document.querySelectorAll("a[href*='item.taobao.com']");
                return links.length > 2;
            }""",
            timeout=timeout_ms,
        )
    except Exception:
        pass
    titles = page.eval_on_selector_all(
        "a[href*='item.taobao.com'], [class*='Card--title'], [class*='title']",
        """els => els.map(e => ({
            txt: (e.innerText || '').trim(),
            href: (e.tagName === 'A' ? e.href : '') || ''
        })).filter(x => x.txt.length > 3)"""
    )
    seen, out = set(), []
    for x in titles:
        t2 = " ".join(x["txt"].split())
        if t2 and t2 not in seen:
            seen.add(t2)
            out.append({"title": t2, "href": x["href"]})
    return out


def _new_context(p):
    browser = p.chromium.launch(headless=False, args=[
        "--disable-blink-features=AutomationControlled",
        "--window-size=1280,900",
    ])
    context = browser.new_context(
        storage_state=STATE_FILE,
        viewport={"width": 1280, "height": 900},
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    )
    return browser, context


def search_image(image_path, candidates=None, max_title=200, browser=None, context=None, retries=2):
    """对单张图片执行淘宝识图，返回聚合统计结果。

    参数 browser/context 用于复用浏览器（批量时传 None 则自建并自动关闭）。
    retries: 淘宝限流/结果未渲染时重试次数（每次重试前冷却）。
    返回 dict:
        name   当选型号（候选/库名映射后，保持识别结果统一）
        model  原始提取到的最高频型号串（未提取则 None）
        votes  该型号标题命中数
        total  抓取到的标题总数
        scores {型号: 命中数}
        titles 前 max_title 条标题
    """
    cands = candidates or DEFAULT_CANDIDATES
    cands = list(dict.fromkeys([str(c) for c in cands]))
    cand_norm = sorted([(norm(c), c) for c in cands], key=lambda x: -len(x[0]))
    _own = browser is None
    if _own:
        with sync_playwright() as p:
            browser, context = _new_context(p)
            try:
                return _search_with_retry(browser, context, image_path, cand_norm, max_title, retries)
            finally:
                browser.close()
    return _search_with_retry(browser, context, image_path, cand_norm, max_title, retries)


def _search_with_retry(browser, context, image_path, cand_norm, max_title, retries):
    """执行并处理限流重试：结果为空或错误时，冷却后重开新页重试。"""
    import time as _t
    for attempt in range(retries + 1):
        r = _search_once(browser, context, image_path, cand_norm, max_title)
        if not r.get("error"):
            return r
        if attempt < retries:
            wait = 20 * (attempt + 1)
            print(f"   [限流/失败重试 {attempt+1}] {r['error']}，冷却 {wait}s...", flush=True)
            _t.sleep(wait)
    return r


def _search_once(browser, context, image_path, cand_norm, max_title):
    page = context.new_page()
    try:
        page.goto(SEARCH_URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)
        # 1) 点击"搜同款/按图片搜索"图标，弹出上传框（拟人第一步）
        icon = page.query_selector("div.search-suggest-image-search-out-icon, div.image-search-icon-outerMode")
        if icon:
            icon.click()
            page.wait_for_timeout(1500)
        # 2) 注入文件（此时 input 已由上传框激活）
        inp = page.query_selector("#image-search-custom-file-input")
        if not inp:
            return {"name": None, "votes": 0, "total": 0, "scores": {}, "titles": [], "error": "no upload input"}
        inp.set_input_files(os.path.abspath(image_path))
        titles = _wait_titles(page)
        if not titles:
            return {"name": None, "votes": 0, "total": 0, "scores": {}, "titles": [], "error": "no results"}
        # 3) 聚合投票：候选（含库名）+ 提取型号，统一按子串出现次数统计
        title_texts = [t["title"] for t in titles]
        scores = Counter()
        for txt in title_texts:
            tn = norm(txt)
            hit = None
            for nc, c in cand_norm:
                if nc in tn:
                    hit = c
                    break  # 每个标题只记一个候选（短候选优先已排序）
            if hit:
                scores[hit] += 1
        # 提取高频型号作为补充候选（新元件/库外型号）
        models = _extract_models(title_texts)
        model_scores = Counter()
        for txt in title_texts:
            best = None
            for m in _MODEL_RE.findall(txt):
                m2 = m.strip("-").strip()
                if not m2 or m2 in _NOISE or len(m2) < 4:
                    continue
                if best is None or len(m2) > len(best):
                    best = m2
            if best:
                model_scores[best] += 1
        # 合并：候选票数 vs 模型票数，取更高者；模型能映射到库名则用库名
        total = len(title_texts)
        name, votes = (None, 0)
        # 英文型号优先：正则提取的英文型号票数 ≥ 2 时直接用英文型号
        all_scores = scores + model_scores
        top_model, top_model_votes = model_scores.most_common(1)[0] if model_scores else (None, 0)
        if top_model and top_model_votes >= 2:
            name, votes = top_model, top_model_votes
        else:
            if all_scores:
                name, votes = all_scores.most_common(1)[0]
        # 中文自动转英文（LED 只支持 ASCII，如 麦克风→MIC、面包板→BREADBOARD）
        name = _cn_to_en(name)
        # 4) 购买链接：取第一条带 href 的结果（综合排序第一位，最相关）
        buy_link = next((t["href"] for t in titles if t.get("href")), "")
        if not buy_link and titles:
            buy_link = f"https://s.taobao.com/search?imageSearch=1"
        return {
            "name": name,
            "model": (models[0][0] if models else None),
            "votes": votes,
            "total": total,
            "scores": dict(all_scores),
            "cand_scores": dict(scores),
            "buy_link": buy_link,
            "titles": [t["title"] for t in titles[:max_title]],
            "error": None,
        }
    finally:
        page.close()


def main():
    ap = argparse.ArgumentParser(description="淘宝识图")
    ap.add_argument("image", help="图片路径")
    ap.add_argument("--candidates", default=None, help="候选型号，逗号分隔")
    ap.add_argument("--titles", action="store_true", help="打印抓到的标题")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print("图片不存在:", args.image)
        sys.exit(1)
    cands = [c.strip() for c in args.candidates.split(",")] if args.candidates else None
    r = search_image(args.image, candidates=cands)
    print(f"图片: {args.image}")
    if r["error"]:
        print("错误:", r["error"])
        sys.exit(1)
    print(f"抓到标题 {r['total']} 条 | 当选: {r['name']} (命中 {r['votes']})")
    print("统计:", json.dumps(r["scores"], ensure_ascii=False))
    if args.titles:
        for i, t in enumerate(r["titles"]):
            print(f"  [{i}] {t}")


if __name__ == "__main__":
    main()
