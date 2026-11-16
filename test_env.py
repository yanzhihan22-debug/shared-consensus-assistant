"""模拟 main.py 环境：先加载 YOLO(torch/ultralytics)，再加载 CLIP，看是否冲突"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

print("[1] 加载 YOLO (模拟 main.py)...")
from ultralytics import YOLO
model = YOLO('yolov8s-seg.pt')
print("   YOLO OK, names:", len(model.names))

print("[2] 加载 CLIP (matcher)...")
import matcher
print("   索引:", {k: v.shape[0] for k, v in matcher.CLIP_INDEX.items()})

print("[3] CLIP 匹配测试...")
db = matcher.load_manual_db()
for key in ['sg90', 'esp32-s3-cam', 'st7789']:
    p = db[key]['photos'][0]
    fp = os.path.join(matcher.BASE_DIR, p.replace('/', os.sep))
    name, score = matcher.match_component(fp)
    print(f"   {key} -> {name} ({score:.3f})")
print("[完成] 无冲突")
