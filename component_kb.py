"""
元件知识库 — PDF 数据手册 + DeepSeek 提取 + 本地缓存
"""
import json
import os
import re
import requests
import fitz  # pymupdf
from dotenv import load_dotenv
load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "components_db.json")
DS_DIR = os.path.join(os.path.dirname(__file__), "datasheets")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def find_pdf(name):
    """搜 datasheets/ 里跟元件名匹配的 PDF"""
    if not os.path.isdir(DS_DIR):
        return None
    name_lower = name.lower().replace(" ", "_").replace("-", "_")
    # 分词
    parts = re.split(r'[_\-\s]+', name_lower)
    
    for f in os.listdir(DS_DIR):
        if not f.lower().endswith(".pdf"):
            continue
        f_lower = f.lower()
        # 文件名包含名称 → 命中
        if name_lower in f_lower:
            return os.path.join(DS_DIR, f)
        # 所有关键词都出现在文件名中 → 命中
        if all(p in f_lower for p in parts if len(p) > 1):
            return os.path.join(DS_DIR, f)
    return None

def read_pdf_text(pdf_path, max_chars=3000):
    """用 pymupdf 读取 PDF 文字内容"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) > max_chars:
                break
        doc.close()
        return text[:max_chars]
    except Exception as e:
        print(f"[PDF] 读取失败: {e}")
        return ""

def extract_from_pdf(pdf_path, component_name):
    """从 PDF 里用 DeepSeek 提取引脚信息"""
    text = read_pdf_text(pdf_path)
    if not text:
        return None
    
    prompt = f"""Extract ONLY the pinout/pin information for: "{component_name}"
from the following datasheet text. Ignore everything else.
Return ONLY valid JSON:
{{
  "name": "{component_name}",
  "voltage": "operating voltage if mentioned",
  "pins": [{{"name": "pin_name", "color": "wire_color", "desc": "function"}}],
  "usage": "brief usage",
  "source": "{os.path.basename(pdf_path)}"
}}

Datasheet text:
{text[:2500]}"""

    resp = requests.post(
        DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.1},
        timeout=20
    )
    if resp.status_code == 200:
        content = resp.json()["choices"][0]["message"]["content"]
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except:
                pass
    return None

def lookup_and_cache(name, qwen_raw=""):
    """查本地库 → 搜PDF → DeepSeek提取 → 缓存"""
    db = load_db()
    key = name.strip().lower().replace(" ", "_")[:40]
    
    # 1. 本地库命中
    if key in db:
        print(f"[知识库] 缓存命中: {db[key]['name']}")
        return db[key]
    
    # 2. Qwen模糊匹配老条目
    if qwen_raw:
        for k, comp in db.items():
            for kw in comp.get('keywords', []):
                if kw.lower() in qwen_raw.lower() or qwen_raw.lower() in kw.lower():
                    print(f"[知识库] Qwen匹配: {comp['name']}")
                    return comp
    
    # 3. 搜 datasheets/ 里的 PDF
    pdf_path = find_pdf(name)
    if pdf_path:
        print(f"[PDF] 找到: {os.path.basename(pdf_path)}")
        result = extract_from_pdf(pdf_path, name)
        if result:
            keywords = [name.lower()] + name.lower().split()
            if qwen_raw: keywords.extend(qwen_raw.lower().split())
            result["keywords"] = keywords
            result["pdf"] = os.path.basename(pdf_path)
            db[key] = result
            save_db(db)
            print(f"[PDF] 已入库: {result['name']}")
            return result
    
    # 4. 兜底：DeepSeek 通用知识
    print(f"[知识库] 无PDF，通用搜索: {name}")
    result = _agent_fallback(name)
    if result:
        keywords = [name.lower()] + name.lower().split()
        if qwen_raw: keywords.extend(qwen_raw.lower().split())
        result["keywords"] = keywords
        db[key] = result
        save_db(db)
        print(f"[知识库] Agent入库: {result['name']}")
    return result

def _agent_fallback(name):
    """DeepSeek 通用知识兜底"""
    prompt = f"""Search your knowledge for: "{name}". Return ONLY valid JSON:
{{
  "name": "full part name",
  "voltage": "operating voltage",
  "pins": [{{"name": "pin", "color": "color", "desc": "function"}}],
  "usage": "brief usage"
}}"""
    resp = requests.post(
        DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.1},
        timeout=20
    )
    if resp.status_code == 200:
        content = resp.json()["choices"][0]["message"]["content"]
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except:
                pass
    return None
