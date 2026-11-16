"""
后端元件匹配 — CLIP 图像相似度 + 丝印文字 匹配 MANUAL_DB.json
仅用于后端识别，不碰前端共识。
"""
import os
import json
import numpy as np

# 使用国内镜像 + 禁用 Xet，避免每次加载权重时联网检查 safetensors
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "MANUAL_DB.json")

CONFIDENT_THRESHOLD = 0.8  # 高于此值直接确定；否则转 Qwen 兜底

_model = None
_preprocess = None

def load_manual_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_model():
    """懒加载 CLIP 模型"""
    global _model, _preprocess
    if _model is None:
        import open_clip
        _model, _preprocess, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k')
        _model.eval()
    return _model, _preprocess

def _load_image(path):
    """读图 → RGB PIL → 预处理"""
    from PIL import Image
    _, preprocess = _get_model()
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0)

def _encode(path):
    """单图 → 归一化向量"""
    import torch
    model, _ = _get_model()
    with torch.no_grad():
        x = _load_image(path)
        emb = model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

def build_index():
    """扫描 MANUAL_DB 所有照片 → {name: [向量,...]}"""
    import torch
    db = load_manual_db()
    index = {}
    for key, entry in db.items():
        vecs = []
        for p in entry.get("photos", []):
            fp = os.path.join(BASE_DIR, p.replace("\\", os.sep))
            if os.path.exists(fp):
                try:
                    vecs.append(_encode(fp))
                except Exception as e:
                    print(f"[匹配] 加载失败 {fp}: {e}")
        if vecs:
            index[key] = np.stack(vecs)
    print(f"[匹配] 索引建立完成: {len(index)} 个元件 / {sum(len(v) for v in index.values())} 张照片")
    return index

def match_query(query_path, index, top_k=3, mode='max'):
    """查询图 → 与库内照片算余弦相似度 → 返回 [(name, score, photo), ...]
    mode='max'：取类内最高分；mode='avg'：取类内平均分（照片多的类不占优）"""
    if not index:
        return []
    q = _encode(query_path)
    results = []
    for key, vecs in index.items():
        sims = vecs @ q
        if mode == 'avg':
            best = float(np.mean(sims))
            best_photo = None
        else:
            best = float(np.max(sims))
            best_photo = index_photos.get(key, [None])[int(np.argmax(sims))] if key in index_photos else None
        results.append((key, best, best_photo))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def match_library(query_path, index=None, min_abs=0.62, min_gap=0.05, mode='max'):
    """库内匹配：返回 (name, score, gap, reliable)。
    用类内最高分（照片多的类仍公平，因为取该类的代表性最佳匹配）。
    reliable 判定：分数足够高（>=0.75），或 top1 领先 top2（gap>=min_gap）且分数>=min_abs。
    0.62/0.05 是针对小库 + 同元件多状态照片的宽容阈值：
    库内元件外观差异大，0.62+ 且领先第二名即视为可信。
    """
    if index is None:
        index = CLIP_INDEX
    results = match_query(query_path, index, top_k=2, mode=mode)
    if not results:
        return None, 0.0, 0.0, False
    name, score, _ = results[0]
    gap = score - (results[1][1] if len(results) > 1 else 0.0)
    reliable = score >= 0.75 or (score >= min_abs and gap >= min_gap)
    return name, score, gap, reliable

def match_component(query_path, index=None, threshold=CONFIDENT_THRESHOLD):
    """识别入口：命中且超过阈值 → 返回 (name, score)；未命中 → (None, score) 转 Qwen 兜底"""
    if index is None:
        index = CLIP_INDEX
    results = match_query(query_path, index, top_k=1)
    if not results:
        return None, 0.0
    name, score, _ = results[0]
    if score >= threshold:
        return name, score
    return None, score

def lookup_by_name(name):
    """按型号名查 MANUAL_DB"""
    db = load_manual_db()
    key = name.strip().lower().replace(" ", "_")
    return db.get(key)

index_photos = {}

def load_index():
    """加载索引 + 记录每张照片路径（供调试显示）"""
    global index_photos
    db = load_manual_db()
    index_photos = {}
    for key, entry in db.items():
        index_photos[key] = entry.get("photos", [])
    return build_index()

# 模块加载时自动建索引（GPU 上约几秒）
try:
    CLIP_INDEX = load_index()
except Exception as e:
    print(f"[匹配] 索引初始化失败: {e}")
    CLIP_INDEX = {}
