import cv2
import numpy as np
import torch
import mediapipe as mp
import sys
import os
from dotenv import load_dotenv
load_dotenv()
from collections import deque, Counter
import serial
import socket
import time
import math
import dashscope
from dashscope import MultiModalConversation
import base64
import threading
import queue
import json
import re
import requests
from voice_control import run_voice_control
from component_kb import lookup_and_cache

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# 持久聊天记忆（Agent 会记住之前说过的元件）
agent_chat = [{"role": "system", "content": "You are a desktop electronics assistant. "
               "User will tell you components one by one (with pin info). Remember each. "
               "When user says 'tell wiring', suggest connections between all remembered components. "
               "Be short. Use | to separate lines. Max 19 chars per line. "
               "USE ENGLISH ONLY. Use ASCII characters only. No Chinese."}]

def _ask_agent(text):
    """给 Agent 发消息，带回记忆"""
    global agent_chat, wiring_advice, show_wiring
    agent_chat.append({"role": "user", "content": text})
    try:
        resp = requests.post(DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "messages": agent_chat, "temperature": 0.1}, timeout=25)
        if resp.status_code != 200:
            # 503 = DeepSeek 服务端过载，明确提示而非静默
            print(f"【Agent错误】HTTP {resp.status_code}: {resp.text[:120]}"
                  + ("（DeepSeek 服务过载，稍后重试）" if resp.status_code == 503 else ""))
            return None
        reply = resp.json()["choices"][0]["message"]["content"]
        agent_chat.append({"role": "assistant", "content": reply})
        raw = reply.replace('\n', '|').replace('\r', '')
        # 过滤非 ASCII 字符（LCD 不支持中文）
        wiring_advice = ''.join(c for c in raw if ord(c) < 128)[:200]
        print(f"【Agent回复】{wiring_advice[:120]}")
        return reply
    except Exception as e:
        print(f"【Agent错误】{e}")
    return None

def _add_component(detail):
    """识别到一个元件，告诉Agent让它记住"""
    global component_list
    if not detail:
        return
    name = detail.get('name', '')
    # 避免重复
    for c in component_list:
        if c.get('name') == name:
            return
    component_list.append(detail)
    # 告诉 Agent：用引脚信息构造消息
    pins_str = ""
    for p in detail.get('pins', []):
        pins_str += f"{p.get('name','')} ({p.get('color','')}): {p.get('desc','')}\n"
    msg = f"我找到了元件: {name}\n引脚信息:\n{pins_str}\n先记住这个元件，后面我还会添加其他元件。"
    print(f"【告诉Agent】{name}")
    _ask_agent(msg)

def _trigger_wiring():
    """接到『接线模式』指令时：开启接线模式，清空旧元件列表。
    后续每次识别自动匹配本地库并加入记忆。
    不需要先识别过任何物体。"""
    global component_list, wiring_advice, last_wiring_count, show_wiring, wiring_mode, component_cache
    component_list = []
    component_cache = None
    wiring_advice = ""
    last_wiring_count = 0
    wiring_mode = True
    show_wiring = False
    print("【接线模式】已开启，后续识别自动匹配引脚并记住元件")
    return True

def _full_reset():
    """完全重置到初始状态 — C键 和 语音清空 共用"""
    global component_list, wiring_advice, show_wiring, last_wiring_count
    global agent_chat, wiring_mode
    global freeze_mode, freeze_img, freeze_img_paths
    global identifying, identify_result, candidates, component_cache
    global qwen_raw, pending_hand_crop, use_hand_prompt, manual_input
    global last_identified, buy_link, last_ser_msg
    global lock_active, lock_tid, lock_hand_center, lock_confirm_count
    global pending_switch_tid, pending_switch_count
    global shake_lock_active, shake_lock_tid, shake_lock_label
    global shake_lock_slot, shake_lock_anchor, locked_obj_persistent
    global cached_objects, cached_person_data, object_tracker
    global consensus_history, last_consensus_label, consensus_hold_counter
    global prev_anchors, hand_positions, hand_shaking, shake_hold_counters
    global need_prompt

    component_list = []; wiring_advice = ""; show_wiring = False; last_wiring_count = 0
    agent_chat = [{"role": "system", "content": "You are a desktop electronics assistant. "
                  "User will tell you components one by one (with pin info). Remember each. "
                  "When user says 'tell wiring', suggest connections between all remembered components. "
                  "Be short. Use | to separate lines. Max 19 chars per line. "
                  "USE ENGLISH ONLY. Use ASCII characters only. No Chinese."}]
    wiring_mode = False
    freeze_mode = False; freeze_img = None; freeze_img_paths = None
    identifying = False; identify_result = ""; candidates = []; component_cache = None
    qwen_raw = ""; pending_hand_crop = False; use_hand_prompt = False; manual_input = ""
    last_identified = ""; buy_link = ""; last_ser_msg = ""
    lock_active = False; lock_tid = None; lock_hand_center = None; lock_confirm_count = 0
    pending_switch_tid = None; pending_switch_count = 0
    shake_lock_active = False; shake_lock_tid = None; shake_lock_label = None
    shake_lock_slot = -1; shake_lock_anchor = None; locked_obj_persistent = None
    cached_objects = []; cached_person_data = []
    object_tracker.clear(); object_tracker['_next_id'] = 0
    consensus_history.clear(); last_consensus_label = None; consensus_hold_counter = 0
    prev_anchors = [None, None]
    hand_positions = [deque(maxlen=15), deque(maxlen=15)]
    hand_shaking = [False, False]; shake_hold_counters = [0, 0]
    need_prompt = False
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("【完全重置】状态已清空")

def _feedback_photo(src_path, name, score=None):
    """识别成功 → 把真实拍摄截图回填进元件库（越用越准），并同步 CLIP 索引
    防污染：仅当 CLIP 置信度足够高（>=FEEDBACK_MIN_SCORE）且该图与目标类自洽时才回填。
    低置信识别只出型号，绝不回填，避免 ST7789 误判污染 ESP32 库。"""
    try:
        import matcher
        if score is None or score < FEEDBACK_MIN_SCORE:
            print(f"【回填跳过】置信度 {score if score is not None else '未知'} < {FEEDBACK_MIN_SCORE}，不污染库")
            return
        name = name.strip()
        key = name.lower().replace(" ", "_")
        # 自洽校验：该图在现有库中最像的必须是目标类
        try:
            chk_name, chk_score, chk_gap, chk_ok = matcher.match_library(src_path)
            chk_key = (chk_name or '').lower().replace(" ", "_")
            if chk_key != key:
                print(f"【回填跳过】该图最像 {chk_name} ({chk_score:.2f})，与目标 {name} 不一致")
                return
        except Exception:
            pass
        folder = os.path.join(matcher.BASE_DIR, 'ref_images', name)
        os.makedirs(folder, exist_ok=True)
        ts = int(time.time())
        dst = os.path.join(folder, f'cam_{ts}.jpg')
        if os.path.abspath(src_path) != os.path.abspath(dst):
            img = cv2.imread(src_path)
            if img is not None:
                cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                import json
                db = matcher.load_manual_db()
                if key not in db:
                    db[key] = {"name": name, "pins": [], "photos": []}
                rel = os.path.relpath(dst, matcher.BASE_DIR).replace('\\', '/')
                if rel not in db[key].get('photos', []):
                    db[key]['photos'].append(rel)
                    with open(matcher.DB_PATH, 'w', encoding='utf-8') as f:
                        json.dump(db, f, ensure_ascii=False, indent=2)
                    matcher.CLIP_INDEX = matcher.load_index()
                    print(f"【回填】已保存实拍照片到库: {name}")
    except Exception as e:
        print(f"【回填失败】{e}")

# 阿里云通义千问 VL 配置
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")
def taobao_identify(image_paths):
    """识别元件：淘宝识图（拍立淘）直接识别锁定截图。
    只输出型号 + 购买链接；接线模式下自动记住元件。"""
    global identify_result, identifying, candidates, buy_link, last_identified, wiring_mode, component_cache
    identifying = True
    identify_result = ""
    candidates = []
    buy_link = ""
    try:
        import taobao_search
        if not image_paths:
            raise ValueError("无截图")
        img = image_paths[0]
        print(f"【淘宝识图】{img}")
        r = taobao_search.search_image(img)
        if r.get("error"):
            print(f"【淘宝识图失败】{r['error']}")
            identify_result = "未识别"
            identifying = False
            return identify_result
        name = r.get("name")
        if name:
            identify_result = name
            last_identified = name
            buy_link = r.get("buy_link", "")
            print(f"【淘宝识别】{name} (votes={r.get('votes')}/{r.get('total')})")
            print(f"【购买链接】{buy_link}")
            # 接线模式：每次识别自动匹配引脚 + 加入元件记忆
            if wiring_mode:
                detail = lookup_and_cache(name, "")
                if detail:
                    component_cache = detail
                    _add_component(detail)
                    print(f"【接线自动加入】{name}")
        else:
            print(f"【淘宝未识别】抓取 {r.get('total')} 条标题均无匹配")
            identify_result = "未识别"
    except Exception as e:
        print(f"【淘宝识图异常】{e}")
        identify_result = "Error"
    identifying = False
    return identify_result

# ==========================================================
# 1. 加载 YOLOv8s-seg 分割模型
# ==========================================================
print("【系统日志】正在加载 yolov8s-seg.pt 分割模型...")
from ultralytics import YOLO
model = YOLO('yolov8s-seg.pt')
names = model.names


def parse_yolo(results, frame_shape):
    """解析 ultralytics YOLO 输出为统一格式"""
    fh, fw = frame_shape[:2]
    objects = []
    person_data = []
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        has_masks = r.masks is not None
        num_boxes = len(boxes)
        num_cls = len(cls_ids)
        for i in range(num_boxes):
            if i >= num_cls:
                continue
            xmin, ymin, xmax, ymax = boxes[i]
            cid = int(cls_ids[i])
            if isinstance(names, dict):
                label = names.get(cid)
            elif cid < len(names):
                label = names[cid]
            else:
                continue
            if label is None:
                continue
            if label == 'person' and has_masks and i < len(r.masks.data):
                mask_np = r.masks.data[i].cpu().numpy()
                mask = cv2.resize(mask_np, (fw, fh), interpolation=cv2.INTER_NEAREST)
                person_data.append({'mask': mask, 'box': [int(xmin), int(ymin), int(xmax), int(ymax)]})
                continue
            if label in ('person', 'dining table', 'chair', 'couch', 'bed'):
                continue
            mask = None
            if has_masks and i < len(r.masks.data):
                mask_np = r.masks.data[i].cpu().numpy()
                mask = cv2.resize(mask_np, (fw, fh), interpolation=cv2.INTER_NEAREST)
            objects.append({
                'label': label,
                'box': [int(xmin), int(ymin), int(xmax), int(ymax)],
                'mask': mask,
            })
    return objects, person_data


def _extract_hand_from_person_mask(person_mask, person_box, h, w):
    """从 YOLO 人体 mask 中用形态学提取手部（MediaPipe 兜底）"""
    x1, y1, x2, y2 = person_box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[y1:y2, x1:x2] = (person_mask[y1:y2, x1:x2] > 0.5).astype(np.uint8)

    kernel_size = max(15, min(h, w) // 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    torso = cv2.erode(mask_full, kernel, iterations=2)
    arms_region = cv2.subtract(mask_full, torso)

    contours, _ = cv2.findContours(arms_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    arm_items = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 300:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        ccx = int(M['m10'] / M['m00'])
        ccy = int(M['m01'] / M['m00'])
        dist = ((ccx - cx)**2 + (ccy - cy)**2)**0.5
        if dist < 30:
            continue
        arm_items.append((cnt, ccx, ccy, dist))

    if not arm_items:
        return None

    arm_items.sort(key=lambda x: x[3], reverse=True)
    best_contour, _, _, _ = arm_items[0]

    farthest_dist = 0
    tip_x, tip_y = cx, cy
    for pt in best_contour:
        px, py = pt[0]
        d = ((px - cx)**2 + (py - cy)**2)**0.5
        if d > farthest_dist:
            farthest_dist = d
            tip_x, tip_y = px, py

    # 沿轮廓往回缩，找手腕处（轮廓宽度变窄的地方）
    wrist_dist = farthest_dist * 0.6  # 取指尖到中心的 60% 处作为手腕
    wx, wy = tip_x, tip_y
    for pt in best_contour:
        px, py = pt[0]
        d = ((px - cx)**2 + (py - cy)**2)**0.5
        if abs(d - wrist_dist) < farthest_dist * 0.1:
            wx, wy = px, py
            break

    hand_radius = int(((tip_x - wx)**2 + (tip_y - wy)**2)**0.5 * 0.55)
    hand_radius = max(20, min(hand_radius, 50))

    hx1 = max(0, tip_x - hand_radius)
    hy1 = max(0, tip_y - hand_radius)
    hx2 = min(w, tip_x + hand_radius)
    hy2 = min(h, tip_y + hand_radius)

    h_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(h_mask, (tip_x, tip_y), hand_radius, 1, -1)

    return {
        'finger_tip': (tip_x, tip_y),
        'hand_box': [hx1, hy1, hx2, hy2],
        'palm': (tip_x, tip_y),
        'gesture': 'POINTING',
        'landmarks': None,
        'source': 'yolo_fallback',
        'anchor': (tip_x, tip_y),
        'anchor_color': (255, 0, 255),
        'anchor_label': 'ARM',
        'hand_mask': h_mask,
    }


def _detect_shake(pos_history, var_thresh=20, zero_thresh=2):
    """检测手部抖动：方差大 + 方向过零点多 = 抖动"""
    if len(pos_history) < 5:
        return False
    xs = [p[0] for p in pos_history]
    ys = [p[1] for p in pos_history]
    var_x = np.var(xs)
    var_y = np.var(ys)
    zero_x = 0
    for i in range(len(xs) - 2):
        if (xs[i+1] - xs[i]) * (xs[i+2] - xs[i+1]) < 0:
            zero_x += 1
    zero_y = 0
    for i in range(len(ys) - 2):
        if (ys[i+1] - ys[i]) * (ys[i+2] - ys[i+1]) < 0:
            zero_y += 1
    return (var_x + var_y) > var_thresh and (zero_x + zero_y) > zero_thresh


def _detect_shake_strong(pos_history):
    """抖动检测（更严格，用于锁切换）"""
    if len(pos_history) < 8:
        return False
    xs = [p[0] for p in pos_history]
    ys = [p[1] for p in pos_history]
    var_x = np.var(xs)
    var_y = np.var(ys)
    total_dist = sum(((xs[i+1]-xs[i])**2 + (ys[i+1]-ys[i])**2)**0.5 for i in range(len(xs)-1))
    zero_x = 0
    for i in range(len(xs) - 2):
        if (xs[i+1] - xs[i]) * (xs[i+2] - xs[i+1]) < 0:
            zero_x += 1
    zero_y = 0
    for i in range(len(ys) - 2):
        if (ys[i+1] - ys[i]) * (ys[i+2] - ys[i+1]) < 0:
            zero_y += 1
    return (var_x + var_y) > 50 and total_dist > 80 and (zero_x + zero_y) > 3


# ==========================================================
# 2. 初始化 MediaPipe 交互手势模块
# ==========================================================
print("【系统日志】正在初始化 MediaPipe...")
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        model_complexity=1, min_detection_confidence=0.3,
                        min_tracking_confidence=0.3)

# ==========================================================
# 3. 唤醒摄像头
# ==========================================================
cap = None
for cam_id in (0, 1, 2):
    print(f"【系统日志】正在尝试摄像头 ID={cam_id}...")
    test = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not test.isOpened():
        test = cv2.VideoCapture(cam_id)
    if test.isOpened():
        # MJPG + 720p：外接 USB 摄像头（ID=0）支持 1280x720@30，内置不支持时自动回退
        test.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        test.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        test.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        test.set(cv2.CAP_PROP_FPS, 30)
        ret, _ = test.read()
        if ret:
            cap = test
            print(f"【成功】使用摄像头 ID={cam_id} ({cap.get(3):.0f}x{cap.get(4):.0f} @ {cap.get(5):.0f}fps)")
            break
    test.release()

if cap is None:
    print("【错误】未能成功打开任何摄像头，请检查硬件！")
    sys.exit()

print("\n>>> 视觉系统完全启动成功！按下键盘上的 'q' 键可退出。")

# ==========================================================
# 4a. 初始化串口（ESP32 USB 通信）
# ==========================================================
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM4")
SERIAL_BAUD = 115200
ser = None
try:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
    print(f"【成功】串口 {SERIAL_PORT} 已打开（波特率 {SERIAL_BAUD}）")
except Exception as e:
    print(f"【警告】串口 {SERIAL_PORT} 不可用：{e}（无串口传输继续运行）")

# ==========================================================
# 4b. 初始化 UDP（WiFi → ESP32 LCD，烧录后查看 LCD 上的 IP）
# ==========================================================
ESP32_IP = os.getenv("ESP32_IP", "192.168.1.200")  # ESP32 固定 IP，可用环境变量覆盖
ESP32_PORT = 8888
sock = None
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"【成功】UDP 目标 {ESP32_IP}:{ESP32_PORT}")
except Exception as e:
    print(f"【警告】UDP 初始化失败：{e}（无 UDP 传输继续运行）")

# ==========================================================
# 4c. 语音控制（暂关闭，测试时用键盘）
# ==========================================================
cmd_queue = queue.Queue()
VOICE_ENABLED = True
if VOICE_ENABLED:
    try:
        vt = threading.Thread(target=run_voice_control, args=(cmd_queue,), daemon=True)
        vt.start()
        print("【语音】语音控制线程已启动")
    except Exception as e:
        print(f"【警告】语音控制启动失败：{e}")
else:
    print("【语音】已关闭（测试模式）")

# ==========================================================
# 5. 实时核心交互循环
# ==========================================================

SKIP_FRAMES = 1
frame_count = 0
cached_objects = []
cached_person_data = []

consensus_history = deque(maxlen=8)
object_tracker = {'_next_id': 0}

last_consensus_label = None
consensus_hold_counter = 0

lock_active = False
lock_tid = None
lock_hand_center = None
LOCK_MOVE_THRESH = 120
lock_confirm_count = 0
pending_switch_tid = None
pending_switch_count = 0

# 晃动锁定（双手二选一后的持久锁定）
shake_lock_active = False
shake_lock_tid = None
shake_lock_label = None  # 用 label 而非 TID 匹配（TID 会变）
shake_lock_slot = -1
shake_lock_anchor = None
SHAKE_LOCK_MOVE_THRESH = 120

# 持久化锁定物体（锁定后即使 YOLO 跟丢也保持位置）
locked_obj_persistent = None  # {'box':[x1,y1,x2,y2], 'frame':frame} 或 None

# 冻结 & 识别状态
freeze_mode = False        # 是否在冻结模式
freeze_img = None          # 冻结时的裁剪图 (numpy)
freeze_img_paths = None    # 保存的文件路径列表 [crop, full]
identifying = False        # 正在请求 API
identify_result = ""       # API 返回结果
candidates = []            # 候选型号列表
component_cache = None     # 本次识别到的详细信息
buy_link = ""              # 淘宝识别到的购买链接
last_identified = ""       # 最近一次识别到的型号（供接线触发用）
wiring_mode = False        # 接线模式开关：开启后每次识别自动记住元件
component_list = []        # 持久列表，跨冻结累积识别到的元件
wiring_advice = ""         # Agent 给出的接线建议
show_wiring = False         # 是否在 LCD 显示接线建议（w 键切换）
last_wiring_count = 0       # 上次生成接线建议时的元件数，有新元件自动重新生成
manual_input = ""          # 手动输入型号名累积
agent_chat = [{"role": "system", "content": "You are a desktop electronics assistant. "
               "User will tell you components one by one (with pin info). Remember each. "
               "When user says 'tell wiring', suggest connections between all remembered components. "
               "Be short. Use | to separate lines. Max 19 chars per line. "
               "USE ENGLISH ONLY. Use ASCII characters only. No Chinese."}]
qwen_raw = ""             # Qwen VL 原始输出，存为错误匹配关键词
pending_hand_crop = False  # 空格确认：YOLO 未检测到时对手指周围截图
use_hand_prompt = False    # 使用手部截图提示词

# 双手状态：每只手独立 anchor 平滑 + 位置历史
prev_anchors = [None, None]
hand_positions = [deque(maxlen=15), deque(maxlen=15)]
hand_shaking = [False, False]
shake_hold_counters = [0, 0]
SHAKE_HOLD = 5  # 抖动检测后保持 True 的帧数（防闪烁）
need_prompt = False  # 需要用户抖动确认

# 双舵机云台手部跟踪
tracking_enabled = True  # 云台跟踪默认开启（t 键开关）
S1_CENTER = 50; S2_CENTER = 120  # S1 默认略向上（向下看桌面的角度收一点）
last_s1 = float(S1_CENTER); last_s2 = float(S2_CENTER)
TRACK_SMOOTH = 0.15  # 平滑系数（越小越平滑）
last_track_send = 0
TRACK_INTERVAL = 0.05  # 发送间隔 50ms
TRACK_DEADZONE = 3.0   # 死区角度（目标与当前差小于此值不追）
# 稳定保持：手稳定后锁住舵机，防止左右摆
hold_s1 = None; hold_s2 = None
hold_count = 0
HOLD_STABLE_FRAMES = 10   # 连续稳定帧数后锁定
HOLD_UNLOCK_ANGLE = 6.0   # 目标偏离锁定值超过此值才解锁

# FPS 计数
fps_counter = 0
fps_timer = cv2.getTickCount()
fps_current = 0.0

consensus_log = deque(maxlen=10)

try:
    last_ser_msg = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("【错误】无法接收摄像头画面帧！尝试重新打开...")
            cap.release()
            cap = None
            for cam_id in (0, 1, 2):
                test = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
                if not test.isOpened():
                    test = cv2.VideoCapture(cam_id)
                if test.isOpened():
                    test.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    test.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    test.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    test.set(cv2.CAP_PROP_FPS, 30)
                    ret, _ = test.read()
                    if ret:
                        cap = test
                        print(f"【恢复】摄像头 ID={cam_id} 已重新打开")
                        break
                test.release()
            if cap is None:
                print("【错误】无法重新打开摄像头，退出。")
                break
            continue

        frame = cv2.flip(frame, 1)
        raw_frame = frame.copy()
        h, w, c = frame.shape

        # ------------------------------------------------------
        # MediaPipe 手部追踪（检测所有手）
        # ------------------------------------------------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = hands.process(rgb_frame)

        multi_hand_data = []
        roi_box = None

        if mp_results.multi_hand_landmarks:
            raw_hands = []
            for hld in mp_results.multi_hand_landmarks:
                xs = [lm.x * w for lm in hld.landmark]
                ys = [lm.y * h for lm in hld.landmark]
                finger_tip = hld.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                ft_x, ft_y = int(finger_tip.x * w), int(finger_tip.y * h)
                hand_box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                palm_x = int((xs[0] + xs[1] + xs[5] + xs[17]) / 4)
                palm_y = int((ys[0] + ys[1] + ys[5] + ys[17]) / 4)
                tip_idx = [4, 8, 12, 16, 20]
                tip_dists = [((xs[i] - palm_x)**2 + (ys[i] - palm_y)**2)**0.5 for i in tip_idx]
                thresh = 0.05 * w
                index_ext = tip_dists[1] > thresh
                middle_ext = tip_dists[2] > thresh
                ring_ext = tip_dists[3] > thresh
                pinky_ext = tip_dists[4] > thresh
                ext_non_thumb = sum([index_ext, middle_ext, ring_ext, pinky_ext])

                if ext_non_thumb >= 4:
                    gesture = "OPEN_PALM"
                elif ext_non_thumb == 0:
                    gesture = "HOLDING"
                elif index_ext:
                    gesture = "POINTING"
                else:
                    gesture = "HOLDING"

                tip_lm = hld.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                mcp_lm = hld.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                wrist_lm = hld.landmark[mp_hands.HandLandmark.WRIST]

                if index_ext:
                    raw_anchor = (int(tip_lm.x * w), int(tip_lm.y * h))
                    anchor_color = (0, 0, 255)
                    anchor_label = 'TIP'
                elif ext_non_thumb <= 1:
                    raw_anchor = (int(mcp_lm.x * w), int(mcp_lm.y * h))
                    anchor_color = (0, 255, 255)
                    anchor_label = 'MCP'
                else:
                    raw_anchor = (int(wrist_lm.x * w), int(wrist_lm.y * h))
                    anchor_color = (255, 0, 0)
                    anchor_label = 'WRIST'

                # 从 landmarks 生成手部 mask
                h_mask = np.zeros((h, w), dtype=np.uint8)
                hull_pts = np.array([(int(lm.x * w), int(lm.y * h)) for lm in hld.landmark], dtype=np.int32)
                hull = cv2.convexHull(hull_pts)
                cv2.fillPoly(h_mask, [hull], 1)
                if index_ext:
                    cv2.circle(h_mask, (int(tip_lm.x * w), int(tip_lm.y * h)), 15, 1, -1)

                raw_hands.append({
                    'finger_tip': (ft_x, ft_y),
                    'hand_box': hand_box,
                    'palm': (palm_x, palm_y),
                    'gesture': gesture,
                    'landmarks': hld,
                    'source': 'mediapipe',
                    'anchor': raw_anchor,
                    'anchor_color': anchor_color,
                    'anchor_label': anchor_label,
                    'hand_mask': h_mask,
                    'index_ext': index_ext,
                })

            # 按 x 坐标排序（左→右），保证 slot 一致性
            raw_hands.sort(key=lambda h: h['hand_box'][0])
            # 合并邻近重复检测（手框中心距离 < 像素阈值则为同一只手）
            merged = []
            for rh in raw_hands:
                cx = (rh['hand_box'][0] + rh['hand_box'][2]) // 2
                cy = (rh['hand_box'][1] + rh['hand_box'][3]) // 2
                dup = False
                for m in merged:
                    mcx = (m['hand_box'][0] + m['hand_box'][2]) // 2
                    mcy = (m['hand_box'][1] + m['hand_box'][3]) // 2
                    if abs(cx - mcx) < 50 and abs(cy - mcy) < 50:
                        dup = True
                        break
                if not dup:
                    merged.append(rh)
            raw_hands = merged[:2]

            for i, hand_dict in enumerate(raw_hands[:2]):
                raw_anchor = hand_dict['anchor']
                if prev_anchors[i] is None:
                    anchor = raw_anchor
                else:
                    anchor = (int(prev_anchors[i][0] * 0.65 + raw_anchor[0] * 0.35),
                              int(prev_anchors[i][1] * 0.65 + raw_anchor[1] * 0.35))
                prev_anchors[i] = anchor
                hand_dict['anchor'] = anchor  # EMA 平滑后的锚点

                # 位置历史用掌心/手框中心（原始检测值，不经过 EMA 平滑）
                box_cx = (hand_dict['hand_box'][0] + hand_dict['hand_box'][2]) // 2
                box_cy = (hand_dict['hand_box'][1] + hand_dict['hand_box'][3]) // 2
                hand_positions[i].append((box_cx, box_cy))

                multi_hand_data.append(hand_dict)

            # 更新抖动状态（带保持计数器：防闪烁）
            for i in range(2):
                if len(hand_positions[i]) >= 5:
                    if _detect_shake(hand_positions[i]):
                        shake_hold_counters[i] = SHAKE_HOLD
                    hand_shaking[i] = shake_hold_counters[i] > 0
                    if shake_hold_counters[i] > 0:
                        shake_hold_counters[i] -= 1

        else:
            # 无 MediaPipe 手 → 重置平滑
            prev_anchors = [None, None]

        # YOLO 兜底：MediaPipe 没检测到时
        if not multi_hand_data and cached_person_data:
            for p in cached_person_data:
                yolo_hand = _extract_hand_from_person_mask(p['mask'], p['box'], h, w)
                if yolo_hand is not None:
                    multi_hand_data.append(yolo_hand)
                    # YOLO 兜底默认使用 slot 0
                    bcx = (yolo_hand['hand_box'][0] + yolo_hand['hand_box'][2]) // 2
                    bcy = (yolo_hand['hand_box'][1] + yolo_hand['hand_box'][3]) // 2
                    hand_positions[0].append((bcx, bcy))
                    hand_shaking[0] = _detect_shake(hand_positions[0])
                    break

        # 设置 hand_data（主手：用于 ROI + 锁 + 调试）
        hand_data = multi_hand_data[0] if multi_hand_data else None

        last_hand_count = len(multi_hand_data)

        # ROI 框：所有手的并集
        if multi_hand_data:
            all_x1 = min(hd['hand_box'][0] for hd in multi_hand_data)
            all_y1 = min(hd['hand_box'][1] for hd in multi_hand_data)
            all_x2 = max(hd['hand_box'][2] for hd in multi_hand_data)
            all_y2 = max(hd['hand_box'][3] for hd in multi_hand_data)
            cx = (all_x1 + all_x2) // 2
            cy = (all_y1 + all_y2) // 2
            half = 320
            rx1 = max(0, cx - half)
            ry1 = max(0, cy - half)
            rx2 = min(w, cx + half)
            ry2 = min(h, cy + half)
            roi_box = (rx1, ry1, rx2, ry2)

        # ------------------------------------------------------
        # YOLOv8-seg 全局检测 + tracker 记忆
        # ------------------------------------------------------
        frame_count += 1
        run_yolo = (frame_count % (SKIP_FRAMES + 1) == 0)

        if run_yolo:
            results = model(frame, verbose=False, device='cuda:0')
            objects_in_frame, persons_in_frame = parse_yolo(results, frame.shape)
            cached_objects = objects_in_frame
            cached_person_data = persons_in_frame
        else:
            objects_in_frame = cached_objects

        fps_counter += 1
        curr_tick = cv2.getTickCount()
        elapsed = (curr_tick - fps_timer) / cv2.getTickFrequency()
        if elapsed >= 1.0:
            fps_current = fps_counter / elapsed
            fps_counter = 0
            fps_timer = curr_tick

        objects_in_frame = [obj for obj in objects_in_frame if obj['label'] != 'person']

        # ------------------------------------------------------
        # 实例级跟踪
        # ------------------------------------------------------
        matched_tids = set()
        for obj in objects_in_frame:
            label = obj['box']
            label = obj['label']
            dcx = (obj['box'][0] + obj['box'][2]) // 2
            dcy = (obj['box'][1] + obj['box'][3]) // 2
            best_id = None
            best_dist = 80
            for tid, tdata in object_tracker.items():
                if not isinstance(tdata, dict) or tdata.get('label') != label:
                    continue
                tcx = (tdata['box'][0] + tdata['box'][2]) // 2
                tcy = (tdata['box'][1] + tdata['box'][3]) // 2
                dist = ((dcx - tcx)**2 + (dcy - tcy)**2)**0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is not None:
                object_tracker[best_id]['box'] = obj['box']
                object_tracker[best_id]['age'] = 0
                if obj.get('mask') is not None:
                    object_tracker[best_id]['mask'] = obj['mask']
                matched_tids.add(best_id)
            else:
                tid = object_tracker['_next_id']
                object_tracker['_next_id'] = tid + 1
                obj_data = {'label': label, 'box': obj['box'], 'age': 0}
                if obj.get('mask') is not None:
                    obj_data['mask'] = obj['mask']
                object_tracker[tid] = obj_data
                matched_tids.add(tid)

        for tid in list(object_tracker):
            if tid == '_next_id':
                continue
            if tid not in matched_tids:
                object_tracker[tid]['age'] += 1
                if object_tracker[tid]['age'] > 15:
                    del object_tracker[tid]

        tracked_objects = []
        for tid, data in object_tracker.items():
            if tid == '_next_id':
                continue
            if data['age'] <= 10:
                obj = {'label': data['label'], 'box': data['box'], 'tid': tid}
                if data.get('mask') is not None:
                    obj['mask'] = data['mask']
                tracked_objects.append(obj)

        label_counter = {}
        for obj in tracked_objects:
            lbl = obj['label']
            if lbl not in label_counter:
                instances = [o for o in tracked_objects if o['label'] == lbl]
                if len(instances) > 1:
                    label_counter[lbl] = 0
                    for o in instances:
                        o['display_label'] = f"{lbl}_{label_counter[lbl]}"
                        label_counter[lbl] += 1
                else:
                    instances[0]['display_label'] = lbl
        for obj in tracked_objects:
            if 'display_label' not in obj:
                obj['display_label'] = obj['label']

        # ------------------------------------------------------
        # 每只手独立评分
        # ------------------------------------------------------
        hand_results = []  # [(slot, best_obj, best_score), ...]
        for slot, hdata in enumerate(multi_hand_data):
            hand_mask = hdata.get('hand_mask')
            hx1, hy1, hx2, hy2 = hdata['hand_box']
            hand_pixels = hand_mask.sum() if hand_mask is not None else (hx2 - hx1) * (hy2 - hy1)

            ft_x, ft_y = hdata['finger_tip']

            best_score = 0
            best_obj = None
            for obj in tracked_objects:
                ox1, oy1, ox2, oy2 = obj['box']

                # 指尖必须在物体框内才参与评分
                if not (ox1 <= ft_x <= ox2 and oy1 <= ft_y <= oy2):
                    continue

                overlap = 0
                if hand_mask is not None:
                    iy1 = max(0, oy1)
                    iy2 = min(hand_mask.shape[0], oy2)
                    ix1 = max(0, ox1)
                    ix2 = min(hand_mask.shape[1], ox2)
                    if iy1 < iy2 and ix1 < ix2:
                        overlap = hand_mask[iy1:iy2, ix1:ix2].sum()
                else:
                    ix1 = max(ox1, hx1)
                    iy1 = max(oy1, hy1)
                    ix2 = min(ox2, hx2)
                    iy2 = min(oy2, hy2)
                    if ix1 < ix2 and iy1 < iy2:
                        overlap = (ix2 - ix1) * (iy2 - iy1)
                if overlap > 0:
                    score = 0.5 + 0.5 * min(1.0, overlap / hand_pixels) if hand_pixels > 0 else 1.0
                    if score > best_score:
                        best_score = score
                        best_obj = obj
            if best_score >= 0.5 and best_obj is not None:
                hand_results.append((slot, best_obj, best_score))

        # ------------------------------------------------------
        # 冲突解决：抖动选择 vs 单手
        # ------------------------------------------------------
        consensus_obj = None
        consensus_score_display = 0.0
        need_prompt = False
        shake_selected = False

        if len(hand_results) == 1:
            single_obj = hand_results[0][1]
            single_label = single_obj.get('label')
            if shake_lock_active and shake_lock_label is not None and single_label == shake_lock_label:
                consensus_obj = single_obj
                consensus_score_display = hand_results[0][2]
            else:
                consensus_obj = single_obj
                consensus_score_display = hand_results[0][2]
                shake_lock_active = False
        elif len(hand_results) >= 2:
            obj_by_slot = {slot: (obj, score) for slot, obj, score in hand_results}

            # 晃动锁检查：用 label 匹配（slot 会随手交叉变化，TID 会随 tracker 变化）
            if shake_lock_active and shake_lock_label is not None:
                still_valid = False
                still_from_tracker = False
                new_slot = shake_lock_slot
                for slot, obj, score in hand_results:
                    if obj.get('label') == shake_lock_label:
                        still_valid = True
                        new_slot = slot  # 更新到最新 slot
                        shake_lock_tid = obj.get('tid')  # 更新到最新 TID
                        break
                # tracker 兜底：物体暂被遮挡但还在 tracker 中
                if not still_valid:
                    for tid, tdata in object_tracker.items():
                        if isinstance(tdata, dict) and tdata.get('label') == shake_lock_label and tdata['age'] <= 20:
                            still_valid = True
                            still_from_tracker = True
                            shake_lock_tid = tid
                            # 用 tracker 数据构建临时 consensus（无 hand_results 时保持显示）
                            consensus_obj = {'label': tdata['label'], 'box': tdata['box'], 'tid': tid}
                            if tdata.get('mask') is not None:
                                consensus_obj['mask'] = tdata['mask']
                            consensus_score_display = 0.5
                            break
                # 持久化兜底：tracker 也丢时用锁定时刻的最后一帧
                if not still_valid and locked_obj_persistent is not None:
                    still_valid = True
                    consensus_obj = {'label': shake_lock_label, 'box': locked_obj_persistent['box'],
                                    'tid': shake_lock_tid, 'mask': locked_obj_persistent.get('mask')}
                    consensus_score_display = 0.3
                if still_valid:
                    # 锁靠 tracker 兜底维持，但原锁定手已在 hand_results 里碰着新物体 → 手已主动移开
                    if still_from_tracker and shake_lock_slot in obj_by_slot:
                        still_valid = False
                        shake_lock_active = False
                if still_valid:
                    shake_lock_slot = new_slot
                    cur = hand_positions[shake_lock_slot][-1] if len(hand_positions[shake_lock_slot]) > 0 else None
                    if still_from_tracker:
                        # tracker 恢复时不检查移动（手可能在遮挡中移动）
                        pass
                    elif cur is not None:
                        dx = cur[0] - shake_lock_anchor[0]
                        dy = cur[1] - shake_lock_anchor[1]
                        if dx**2 + dy**2 > SHAKE_LOCK_MOVE_THRESH**2:
                            shake_lock_active = False
                    if shake_lock_active:
                        # 锁定有效：先检查对方手是否要切换
                        other_slot = 1 if shake_lock_slot == 0 else 0
                        switched = False
                        if other_slot in obj_by_slot and _detect_shake_strong(hand_positions[other_slot]):
                            consensus_obj = obj_by_slot[other_slot][0]
                            consensus_score_display = obj_by_slot[other_slot][1]
                            shake_lock_tid = consensus_obj.get('tid')
                            shake_lock_label = consensus_obj.get('label')
                            shake_lock_slot = other_slot
                            shake_lock_anchor = hand_positions[other_slot][-1] if len(hand_positions[other_slot]) > 0 else cur
                            locked_obj_persistent = {'box': consensus_obj['box'],
                                                      'mask': consensus_obj.get('mask'),
                                                      'frame': frame_count}
                            switched = True
                        if not switched:
                            # 保持当前锁定
                            for slot, obj, score in hand_results:
                                if obj.get('tid') == shake_lock_tid:
                                    consensus_obj = obj
                                    consensus_score_display = score
                                    if cur is not None:
                                        shake_lock_anchor = cur
                                    break
                else:
                    shake_lock_active = False  # 手离开物体 → 解锁
                    locked_obj_persistent = None

            # 无晃动锁 → 正常抖动检测（用 hand_shaking 含保持计数器）
            if not shake_lock_active:
                if 0 in obj_by_slot and hand_shaking[0]:
                    if 1 not in obj_by_slot or not hand_shaking[1]:
                        consensus_obj = obj_by_slot[0][0]
                        consensus_score_display = obj_by_slot[0][1]
                        shake_selected = True
                        # 激活晃动锁
                        shake_lock_active = True
                        shake_lock_tid = consensus_obj.get('tid')
                        shake_lock_label = consensus_obj.get('label')
                        shake_lock_slot = 0
                        shake_lock_anchor = hand_positions[0][-1] if len(hand_positions[0]) > 0 else None
                if consensus_obj is None and 1 in obj_by_slot and hand_shaking[1]:
                    if 0 not in obj_by_slot or not hand_shaking[0]:
                        consensus_obj = obj_by_slot[1][0]
                        consensus_score_display = obj_by_slot[1][1]
                        shake_selected = True
                        shake_lock_active = True
                        shake_lock_tid = consensus_obj.get('tid')
                        shake_lock_label = consensus_obj.get('label')
                        shake_lock_slot = 1
                        shake_lock_anchor = hand_positions[1][-1] if len(hand_positions[1]) > 0 else None

            # 锁定中刷新持久化
            if shake_lock_active and consensus_obj is not None:
                locked_obj_persistent = {'box': consensus_obj['box'],
                                          'mask': consensus_obj.get('mask'),
                                          'frame': frame_count}

            # 两只手指向同一物体 → 直接锁定，不用抖动选择
            if not shake_lock_active and len(hand_results) >= 2:
                h0_label = hand_results[0][1].get('label')
                h1_label = hand_results[1][1].get('label')
                if h0_label is not None and h0_label == h1_label:
                    consensus_obj = hand_results[0][1]
                    consensus_score_display = hand_results[0][2]
                    shake_lock_active = True
                    shake_lock_tid = consensus_obj.get('tid')
                    shake_lock_label = consensus_obj.get('label')
                    shake_lock_slot = 0
                    shake_lock_anchor = hand_positions[0][-1] if len(hand_positions[0]) > 0 else None

            # 锁定中就不显示 need_prompt
            if consensus_obj is None:
                need_prompt = True
                consensus_history.clear()  # 清除旧历史，防止时间平滑恢复旧共识

        if len(hand_results) < 2:
            need_prompt = False

        # 记忆共识
        if hand_data and consensus_obj and not shake_selected:
            if not consensus_log or consensus_log[-1]['label'] != consensus_obj['label']:
                consensus_log.append({'label': consensus_obj['label'], 'frame': frame_count})
            last_consensus_label = consensus_obj['label']
            consensus_hold_counter = 30
        elif not hand_data:
            if consensus_hold_counter > 0:
                consensus_hold_counter -= 1
                if last_consensus_label is not None:
                    for tdata in object_tracker.values():
                        if isinstance(tdata, dict) and tdata.get('label') == last_consensus_label:
                            consensus_obj = {'label': last_consensus_label, 'box': tdata['box']}
                            break
            else:
                last_consensus_label = None

        # 时间平滑（晃动锁定时跳过，锁定本身就是稳定机制）
        if shake_lock_active and consensus_obj is not None:
            pass  # 晃动锁定中：不覆盖 consensus_obj
        elif not hand_data:
            consensus_history.clear()
            prev_anchors = [None, None]
        elif consensus_obj:
            consensus_history.append(consensus_obj.get('tid'))
        else:
            consensus_history.append(None)

        if not (shake_lock_active and consensus_obj is not None) and len(consensus_history) == consensus_history.maxlen:
            counter = Counter(consensus_history)
            most_common_tid, count = counter.most_common(1)[0]
            if most_common_tid is not None and count >= 5:
                smoothed = [o for o in tracked_objects if o.get('tid') == most_common_tid]
                if not smoothed:
                    for sid, tdata in object_tracker.items():
                        if sid == most_common_tid:
                            smoothed = [{'label': tdata['label'], 'box': tdata['box'],
                                         'tid': sid, 'mask': tdata.get('mask')}]
                            break
                if smoothed:
                    consensus_obj = smoothed[0]
                else:
                    consensus_obj = None
            else:
                consensus_obj = None

        # ------------------------------------------------------
        # 锁定机制（只在单手时启用）
        # ------------------------------------------------------
        hand_center = None
        if hand_data and len(multi_hand_data) == 1:
            hx1, hy1, hx2, hy2 = hand_data['hand_box']
            hand_center = ((hx1 + hx2) // 2, (hy1 + hy2) // 2)

        if lock_active:
            # 锁定中：YOLO 跟丢时用持久化维持
            if consensus_obj is None and locked_obj_persistent is not None:
                consensus_obj = {'label': 'locked', 'box': locked_obj_persistent['box'],
                                'tid': lock_tid, 'mask': locked_obj_persistent.get('mask')}
            if hand_center is None:
                lock_active = False; locked_obj_persistent = None
            elif lock_hand_center is None:
                lock_hand_center = hand_center
            else:
                dx = hand_center[0] - lock_hand_center[0]
                dy = hand_center[1] - lock_hand_center[1]
                if dx**2 + dy**2 > LOCK_MOVE_THRESH**2:
                    lock_active = False; locked_obj_persistent = None
                else:
                    if consensus_obj:
                        # 刷新持久化
                        locked_obj_persistent = {'box': consensus_obj['box'],
                                                  'mask': consensus_obj.get('mask')}
                        if consensus_obj.get('tid') != lock_tid:
                            new_tid = consensus_obj['tid']
                            if new_tid == pending_switch_tid:
                                pending_switch_count += 1
                                if pending_switch_count >= 5:
                                    lock_tid = new_tid
                                    lock_hand_center = hand_center
                                    lock_confirm_count = 3
                                    pending_switch_tid = None
                                    pending_switch_count = 0
                            else:
                                pending_switch_tid = new_tid
                                pending_switch_count = 1
                        else:
                            pending_switch_tid = None
                            pending_switch_count = 0
        else:
            if hand_data and consensus_obj and len(multi_hand_data) == 1:
                con_tid = consensus_obj.get('tid')
                if con_tid and consensus_score_display >= 0.5:
                    if con_tid == lock_tid:
                        lock_confirm_count = min(lock_confirm_count + 1, 10)
                    else:
                        lock_tid = con_tid
                        lock_confirm_count = 1
                    if lock_confirm_count >= 3:
                        lock_active = True
                        lock_hand_center = hand_center
                        locked_obj_persistent = {'box': consensus_obj['box'],
                                                  'mask': consensus_obj.get('mask'),
                                                  'frame': frame_count}
                elif con_tid and consensus_score_display >= 0.2 and lock_confirm_count > 0:
                    lock_confirm_count -= 1
                else:
                    lock_confirm_count = 0
                    lock_tid = None

        # ------------------------------------------------------
        # 绘制
        # ------------------------------------------------------
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['box']
            label = obj.get('display_label', obj['label'])
            # 检查是否被某只手接触
            touched = False
            for _, h_obj, _ in hand_results:
                if h_obj.get('tid') == obj['tid']:
                    touched = True
                    break
            color = (0, 255, 255) if touched else (0, 255, 0)
            seg_mask = obj.get('mask')
            if seg_mask is not None:
                mask_uint8 = (seg_mask > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if roi_box:
            rx1, ry1, rx2, ry2 = roi_box
            overlay = frame.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # 绘制所有手
        for hi, hdata in enumerate(multi_hand_data):
            if hdata['source'] == 'mediapipe':
                mp_draw.draw_landmarks(frame, hdata['landmarks'], mp_hands.HAND_CONNECTIONS)
            elif hdata['source'] == 'yolo_fallback':
                cv2.putText(frame, 'YOLO HAND',
                            (hdata['hand_box'][0], hdata['hand_box'][1] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            ft_x, ft_y = hdata['finger_tip']
            cv2.circle(frame, (ft_x, ft_y), 6, (0, 0, 255), -1)
            hx1, hy1, hx2, hy2 = hdata['hand_box']
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 1)
            gesture = hdata.get('gesture', '?')
            cv2.putText(frame, gesture, (hx1, hy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            ax, ay = hdata['anchor']
            cv2.circle(frame, (ax, ay), 10, hdata['anchor_color'], -1)
            cv2.putText(frame, hdata['anchor_label'], (ax + 12, ay + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hdata['anchor_color'], 2)

            # 抖动标签
            if hand_shaking[hi] and hi < len(multi_hand_data):
                cv2.putText(frame, 'SHAKE!', (ax - 20, ay - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ----- 云台跟踪 -----
        if tracking_enabled and multi_hand_data:
            ft_x, ft_y = multi_hand_data[0]['finger_tip']
            target_s2 = S2_CENTER + (ft_x - w/2) * (60 / (w/2))
            target_s2 = max(0, min(180, target_s2))
            target_s1 = S1_CENTER + (ft_y - h/2) * (50 / (h/2))
            target_s1 = max(0, min(180, target_s1))
            # 稳定锁定：手不动就锁住，防止左右摆
            if hold_s1 is not None:
                if (abs(target_s1 - hold_s1) < HOLD_UNLOCK_ANGLE and
                        abs(target_s2 - hold_s2) < HOLD_UNLOCK_ANGLE):
                    target_s1 = hold_s1
                    target_s2 = hold_s2
                else:
                    hold_s1 = None; hold_s2 = None  # 手动了，解锁
            else:
                if (abs(target_s1 - last_s1) < TRACK_DEADZONE and
                        abs(target_s2 - last_s2) < TRACK_DEADZONE):
                    hold_count += 1
                    if hold_count >= HOLD_STABLE_FRAMES:
                        hold_s1 = last_s1; hold_s2 = last_s2
                        hold_count = 0
                else:
                    hold_count = 0
            last_s1 = last_s1 * (1 - TRACK_SMOOTH) + target_s1 * TRACK_SMOOTH
            last_s2 = last_s2 * (1 - TRACK_SMOOTH) + target_s2 * TRACK_SMOOTH
            now = time.time()
            if now - last_track_send >= TRACK_INTERVAL:
                last_track_send = now
                cmd = f"s1:{last_s1:.0f} s2:{last_s2:.0f}\n"
                if ser is not None:
                    try: ser.write(cmd.encode())
                    except: pass
                if sock is not None:
                    try: sock.sendto(cmd.encode(), (ESP32_IP, ESP32_PORT))
                    except: pass

        # 共识框
        if consensus_obj:
            ox1, oy1, ox2, oy2 = consensus_obj['box']

            # 抖动选中的添加特殊标记
            if shake_selected:
                label_text = "Consensus: " + consensus_obj['label'] + " [SHAKE]"
                box_color = (0, 255, 255)
            elif shake_lock_active:
                label_text = "Consensus: " + consensus_obj['label'] + " [LOCKED:SHAKE]"
                box_color = (0, 255, 0)
            elif need_prompt:
                label_text = "Consensus: " + consensus_obj['label']
                box_color = (0, 0, 255)
            else:
                label_text = "Consensus: " + consensus_obj['label']
                if not hand_data:
                    label_text += " (Memorized)"
                elif lock_active:
                    label_text += " [LOCKED]"
                box_color = (0, 255, 255) if lock_active else (0, 0, 255)

            cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
            seg_mask = consensus_obj.get('mask')
            if seg_mask is not None and seg_mask.shape[:2] == (h, w):
                mask_uint8 = (seg_mask > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, box_color, 3)
            else:
                cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), box_color, 4)

        # 提示文字（双手都稳时）
        if need_prompt:
            prompt_text = "Both hands stable - Shake the hand of the object you want"
            cv2.putText(frame, prompt_text, (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 调试信息
        debug_text = f"Objects: {len(tracked_objects)}"
        if multi_hand_data:
            hand_count = len(multi_hand_data)
            sources = '+'.join('MP' if h['source'] == 'mediapipe' else 'YOLO' for h in multi_hand_data)
            debug_text += f" | Hands: {hand_count}({sources})"
            for hi in range(hand_count):
                n = len(hand_positions[hi])
                sh = 'S' if hand_shaking[hi] else '-'
                debug_text += f" | H{hi}:{n}{sh}"
            debug_text += f" | Gesture: {multi_hand_data[0]['gesture']}(0)"
            if hand_count > 1:
                debug_text += f"/{multi_hand_data[1]['gesture']}(1)"
        else:
            debug_text += " | Hand: NO"
        debug_text += f" | FPS: {fps_current:.0f}"
        debug_text += f" | Score: {consensus_score_display:.2f}"
        debug_text += f" | Consensus: {consensus_obj['label'] if consensus_obj else 'NONE'}"
        if need_prompt:
            debug_text += " | [PROMPT]"
        if shake_lock_active:
            debug_text += f" | [SHAKE_LOCK:{shake_lock_tid}]"
        if lock_active:
            debug_text += " | [LOCKED]"
        if tracking_enabled:
            debug_text += " | [TRACK]"
        if consensus_hold_counter > 0 and not hand_data:
            debug_text += f" | Hold: {consensus_hold_counter}"
        cv2.putText(frame, debug_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ------------------------------------------------------
        # 冻结显示 & 串口消息
        # ------------------------------------------------------
        ser_msg = "No object"
        # w 切换显示接线建议（按 w 切回来之前一直显示）
        if show_wiring and wiring_advice:
            ser_msg = wiring_advice
        elif freeze_mode:
            if freeze_img is not None:
                fx, fy = 10, h - 180
                disp = cv2.resize(freeze_img, (160, 160))
                frame[fy:fy+160, fx:fx+160] = disp
                if identifying:
                    ser_msg = "Identifying..."
                elif identify_result:
                    if component_cache:
                        name = component_cache.get('name', identify_result)
                        ser_msg = str(name)[:60]
                    else:
                        ser_msg = identify_result
                else:
                    ser_msg = "Collect components..."
                cv2.putText(frame, f"{ser_msg}", (10, fy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif pending_hand_crop:
            ser_msg = "Are you sure? Press SPACE again to capture"
        elif need_prompt:
            ser_msg = "Both hands stable - Shake to select"
        elif consensus_obj:
            if shake_lock_active or lock_active:
                ser_msg = "Consensus ready [LOCKED]"
            else:
                ser_msg = "Consensus ready"

        # ------------------------------------------------------
        # 发送到 LCD（USB 串口 + WiFi UDP），仅内容变化时发送
        # 纯文本协议：\n 结尾，只显示型号（不支持中文/长文本）
        # ------------------------------------------------------
        if ser_msg != last_ser_msg:
            last_ser_msg = ser_msg
            safe_msg = ''.join(c for c in ser_msg if ord(c) < 128)
            payload = (safe_msg + "\n").encode()
            if ser is not None:
                try:
                    ser.write(payload)
                except Exception:
                    pass
            if sock is not None and len(payload) < 256:
                try:
                    sock.sendto(payload, (ESP32_IP, ESP32_PORT))
                except Exception:
                    pass

        cv2.imshow('HRI - Shared Consensus System', frame)

        key = cv2.waitKey(1) & 0xFF

        def _save_crop():
            """截图并保存，返回路径列表（特写 + 全景）"""
            obj = consensus_obj
            if not obj or not (shake_lock_active or lock_active):
                return None
            mask = obj.get('mask')
            box = obj['box']
            x1,y1 = max(0,box[0]),max(0,box[1])
            x2,y2 = min(w,box[2]),min(h,box[3])
            bw, bh = x2 - x1, y2 - y1
            # 40% padding
            pad_x, pad_y = int(bw * 0.4), int(bh * 0.4)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)
            # 有手部数据时：特写用手部+物体联合区域（实拍构图，淘宝识图更准）
            if hand_data:
                hx1, hy1, hx2, hy2 = hand_data['hand_box']
                union_x1 = max(0, min(x1, hx1) - pad_x)
                union_y1 = max(0, min(y1, hy1) - pad_y)
                union_x2 = min(w, max(x2, hx2) + pad_x)
                union_y2 = min(h, max(y2, hy2) + pad_y)
                if (union_x2 - union_x1) > 20 and (union_y2 - union_y1) > 20:
                    cx1, cy1, cx2, cy2 = union_x1, union_y1, union_x2, union_y2

            def _upscale(img, mins=600):
                if img.size == 0: return None
                ch, cw = img.shape[:2]
                if cw < mins or ch < mins:
                    s = max(mins / cw, mins / ch, 1.0)
                    img = cv2.resize(img, (int(cw*s), int(ch*s)), interpolation=cv2.INTER_CUBIC)
                return img

            def _save(img, suffix):
                img = _upscale(img)
                if img is None: return None
                ts = int(time.time())
                d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'consensus_crops')
                os.makedirs(d, exist_ok=True)
                p = os.path.join(d, f'crop_{ts}_{suffix}.jpg')
                cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                return p, img

            # --- 特写：物体突出（mask 压暗背景，扩大 + 放大）---
            crop = raw_frame[cy1:cy2, cx1:cx2].copy()
            # CLIP 用：未压暗的原始裁剪（与库内照片背景一致，相似度更准）
            r0 = _save(crop, 'plain')
            if mask is not None and mask.shape[:2] == (h, w):
                mc = (mask[cy1:cy2, cx1:cx2] > 0).astype(np.uint8)
                dark = cv2.addWeighted(crop, 0.3, np.zeros_like(crop), 0, 0)
                crop = np.where(np.stack([mc]*3, axis=-1), crop, dark)
            r1 = _save(crop, 'crop')

            # --- 全景：完整帧（缩放到最大 800px 宽）---
            full = raw_frame.copy()
            fh, fw = full.shape[:2]
            if fw > 800:
                ratio = 800 / fw
                full = cv2.resize(full, (800, int(fh*ratio)), interpolation=cv2.INTER_CUBIC)
            r2 = _save(full, 'full')

            if r1 is None:
                return None
            paths = []
            if r0:
                paths.append(r0[0])
            paths.append(r1[0])
            if r2:
                paths.append(r2[0])
            return paths, r1[1]

        def _save_hand_crop():
            """对手指周围 200x200 区域截图（YOLO 未检测到时用）"""
            hd = hand_data
            if not hd:
                return None
            ft_x, ft_y = hd['finger_tip']
            s = 200
            hx1 = max(0, ft_x - s//2)
            hy1 = max(0, ft_y - s//2)
            hx2 = min(w, hx1 + s)
            hy2 = min(h, hy1 + s)

            def _up(img, mins=600):
                if img.size == 0: return None
                ch, cw = img.shape[:2]
                if cw < mins or ch < mins:
                    scale = max(mins / cw, mins / ch, 1.0)
                    img = cv2.resize(img, (int(cw*scale), int(ch*scale)), interpolation=cv2.INTER_CUBIC)
                return img

            def _sv(img, suffix):
                img = _up(img)
                if img is None: return None
                ts = int(time.time())
                d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'consensus_crops')
                os.makedirs(d, exist_ok=True)
                p = os.path.join(d, f'hand_{ts}_{suffix}.jpg')
                cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                return p, img

            crop = raw_frame[hy1:hy2, hx1:hx2].copy()
            full = raw_frame.copy()
            fh, fw = full.shape[:2]
            if fw > 800:
                ratio = 800 / fw
                full = cv2.resize(full, (800, int(fh*ratio)), interpolation=cv2.INTER_CUBIC)
            r1 = _sv(crop, 'hand')
            r2 = _sv(full, 'full')
            if r1 is None:
                return None
            paths = [r1[0]]
            if r2:
                paths.append(r2[0])
            return paths, r1[1]

        if key == ord('c') and not manual_input:
            _full_reset()
        if key == ord('w') and not manual_input:
            if not component_list:
                _trigger_wiring()
            n = len(component_list)
            if wiring_advice and n > last_wiring_count:
                print("【新元件】重新查询接线")
                wiring_advice = ""
            if not wiring_advice and n >= 1:
                names = [c.get('name','?')[:10] for c in component_list]
                msg = f"Tell wiring: I have {' + '.join(names)}. How to connect? Use | per line, max 19 chars."
                print(f"【询问接线】{msg}")
                # 后台线程执行，不卡主循环
                t = threading.Thread(target=_ask_agent, args=(msg,), daemon=True)
                t.start()
                last_wiring_count = n
                show_wiring = True
            elif wiring_advice:
                show_wiring = not show_wiring
                print(f"【显示】{'接线建议' if show_wiring else '正常模式'}")
        if key == ord('t') and not manual_input:
            tracking_enabled = not tracking_enabled
            if tracking_enabled:
                print("【跟踪】云台跟踪已开启")
            else:
                hold_s1 = None; hold_s2 = None; hold_count = 0
                print("【跟踪】云台跟踪已关闭")
        # 手动测试舵机（调试用）
        if ser is not None and tracking_enabled:
            if key == ord('1'):
                last_s1 = 0; print("【手动】S1=0")
            elif key == ord('2'):
                last_s1 = 90; print("【手动】S1=90")
            elif key == ord('3'):
                last_s2 = 0; print("【手动】S2=0")
            elif key == ord('4'):
                last_s2 = 180; print("【手动】S2=180")
            if key in (ord('1'), ord('2'), ord('3'), ord('4')):
                cmd = f"s1:{last_s1:.0f} s2:{last_s2:.0f}\n"
                ser.write(cmd.encode())
                print(f"【手动】→ {cmd.strip()}")
        if key == ord('q') and not manual_input:
            break

        # ----- 手动输入模式：优先处理，吃掉所有按键 -----
        if manual_input:
            if key == 13:  # Enter 确认
                detail = lookup_and_cache(manual_input, qwen_raw)
                if detail:
                    component_cache = detail
                    _add_component(detail)
                    identify_result = detail['name']
                else:
                    identify_result = manual_input
                manual_input = ""
                candidates = []
                print(f"【手动入库】{identify_result}")
            elif key == 8:  # Backspace
                manual_input = manual_input[:-1]
                identify_result = manual_input if manual_input else "Type name..."
            elif key == 27:  # Esc 取消
                manual_input = ""
                candidates = []
                identify_result = ""
                print("【取消】手动输入")
            elif key >= 32 and key <= 126:
                manual_input += chr(key)
                identify_result = manual_input
                print(f"[输入] {manual_input}")
            # 不处理其他键
        elif key == 27 and pending_hand_crop:
            pending_hand_crop = False
            print("【取消】手部截图")
        elif key == ord(' ') and not freeze_mode:
            # 空格：进入冻结并锁定
            if consensus_obj is not None:
                # 有 consensus → 正常截图
                if not (shake_lock_active or lock_active):
                    lock_active = True
                r = _save_crop()
                if r:
                    freeze_img_paths, freeze_img = r
                    freeze_mode = True
                    identify_result = ""
                    candidates = []
                    component_cache = None
                    identifying = False
                    pending_hand_crop = False
                    use_hand_prompt = False
                    for _ in range(5):
                        cap.grab()
                    print(f"【冻结】{freeze_img_paths[0]}")
                    # 锁定完成自动识别（CLIP 命中直接出型号，未命中转 Qwen）
                    t = threading.Thread(target=taobao_identify, args=(freeze_img_paths,), daemon=True)
                    t.start()
                    print("【识别】锁定完成，自动识别中...")
            elif pending_hand_crop and hand_data:
                # 第二次空格：对手指周围截图
                use_hand_prompt = True
                r = _save_hand_crop()
                if r:
                    freeze_img_paths, freeze_img = r
                    freeze_mode = True
                    identify_result = ""
                    candidates = []
                    component_cache = None
                    identifying = False
                    pending_hand_crop = False
                    for _ in range(5):
                        cap.grab()
                    print(f"【手部冻结】{freeze_img_paths[0]}")
            elif hand_data:
                # 第一次空格：无 consensus，确认提示
                pending_hand_crop = True
                print("【手部待确认】按空格确认对手指周围截图")
        elif key == ord('m'):
            # M 键：开启接线模式，清空列表，后续识别自动匹配引脚
            _trigger_wiring()
        elif key == ord(' ') and freeze_mode:
            # 空格：退出冻结，同时释放锁定让用户可选新物体
            freeze_mode = False
            freeze_img = None
            freeze_img_paths = None
            identify_result = ""
            candidates = []
            qwen_raw = ""
            component_cache = None
            identifying = False
            pending_hand_crop = False
            use_hand_prompt = False
            shake_lock_active = False
            shake_lock_tid = None
            shake_lock_label = None
            shake_lock_slot = -1
            shake_lock_anchor = None
            lock_active = False
            locked_obj_persistent = None
            print("【解冻】")
        elif key == ord('i') and not identifying:
            # I 键：识别（发 mask 特写 + 全局两张图）
            if freeze_mode and freeze_img_paths:
                t = threading.Thread(target=taobao_identify, args=(freeze_img_paths,), daemon=True)
                t.start()
                print("【识别】正在请求...")
            elif not freeze_mode:
                # 优先截图选中物体，失败则截完整帧
                r = _save_crop()
                if not r:
                    # 无 consensus/lock 时直接截当前画面的完整帧
                    ts = int(time.time())
                    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'consensus_crops')
                    os.makedirs(d, exist_ok=True)
                    p = os.path.join(d, f'crop_{ts}_full.jpg')
                    cv2.imwrite(p, raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    r = ([p], raw_frame)
                if r:
                    freeze_img_paths, freeze_img = r
                    freeze_mode = True
                    identify_result = ""
                    candidates = []
                    component_cache = None
                    identifying = False
                    for _ in range(5):
                        cap.grab()
                    t = threading.Thread(target=taobao_identify, args=(freeze_img_paths,), daemon=True)
                    t.start()
                    print("【识别】截图+请求...")
        elif freeze_mode and candidates and key == 13:
            # Enter: 候选都不对，进入手动输入模式
            candidates = []
            identify_result = ""
            manual_input = ""

        # ----- 手动输入模式 -----
        elif freeze_mode and not identifying and (manual_input or (not candidates and key == 13)):
            if key == 13 and manual_input:  # Enter 确认
                detail = lookup_and_cache(manual_input)
                if detail:
                    component_cache = detail
                    _add_component(detail)
                    identify_result = detail['name']
                else:
                    identify_result = manual_input
                manual_input = ""
                candidates = []
            elif key == 8:  # Backspace
                manual_input = manual_input[:-1]
                identify_result = manual_input if manual_input else "Type name..."
            elif key == 27:  # Esc 取消
                manual_input = ""
                candidates = []
                identify_result = ""
            elif key >= 32 and key <= 126:  # 可打印字符
                manual_input += chr(key)
                identify_result = manual_input
            else:
                # 候选模式：数字选、Enter手输
                pass  # 下面继续处理

        # ----- 候选选择 -----
        elif key >= ord('1') and key <= ord('9') and candidates:
            n = key - ord('1')
            if n < len(candidates):
                chosen = candidates[n]
                print(f"【选择】{n+1}: {chosen}")
                detail = lookup_and_cache(chosen, qwen_raw)
                if detail:
                    component_cache = detail
                    _add_component(detail)
                    identify_result = detail['name']
                else:
                    identify_result = chosen
                candidates = []

        # ------------------------------------------------------
        # 语音命令处理
        # ------------------------------------------------------
        while not cmd_queue.empty():
            cmd = cmd_queue.get_nowait()
            if cmd == "lock":
                if not freeze_mode:
                    if consensus_obj is not None:
                        if not (shake_lock_active or lock_active):
                            lock_active = True
                        r = _save_crop()
                        if r:
                            freeze_img_paths, freeze_img = r
                            freeze_mode = True
                            identify_result = ""
                            identifying = False
                            pending_hand_crop = False
                            use_hand_prompt = False
                            for _ in range(5):
                                cap.grab()
                            # 跟空格键一致：锁定后自动识别
                            t = threading.Thread(target=taobao_identify, args=(freeze_img_paths,), daemon=True)
                            t.start()
                            print("【语音】锁定 + 识图")
                    elif hand_data:
                        pending_hand_crop = True
                        print("【语音】锁定待确认")
                else:
                    print("【语音】已锁定")
            elif cmd == "identify":
                if freeze_mode and freeze_img_paths:
                    t = threading.Thread(target=taobao_identify, args=(freeze_img_paths,), daemon=True)
                    t.start()
                    print("【语音】识别中...")
                elif not freeze_mode:
                    r = _save_crop()
                    if r:
                        freeze_img_paths, freeze_img = r
                        freeze_mode = True
                        identify_result = ""
                        identifying = False
                        for _ in range(5):
                            cap.grab()
                        t = threading.Thread(target=taobao_identify, args=(freeze_img_paths,), daemon=True)
                        t.start()
                        print("【语音】截图+识别...")
            elif cmd == "unlock":
                freeze_mode = False
                freeze_img = None
                freeze_img_paths = None
                identify_result = ""
                identifying = False
                pending_hand_crop = False
                use_hand_prompt = False
                shake_lock_active = False
                shake_lock_tid = None
                shake_lock_label = None
                shake_lock_slot = -1
                shake_lock_anchor = None
                lock_active = False
                locked_obj_persistent = None
                print("【语音】解冻")
            elif cmd == "confirm":
                if pending_hand_crop and hand_data:
                    use_hand_prompt = True
                    r = _save_hand_crop()
                    if r:
                        freeze_img_paths, freeze_img = r
                        freeze_mode = True
                        identify_result = ""
                        identifying = False
                        pending_hand_crop = False
                        for _ in range(5):
                            cap.grab()
                        print("【语音】确认手部截图")
                    else:
                        print("【语音】截图失败")
                else:
                    print("【语音】无需确认")
            elif cmd in ("select_1", "select_2", "select_3") and candidates:
                n = int(cmd[-1]) - 1
                if n < len(candidates):
                    chosen = candidates[n]
                    print(f"【语音】选择: {chosen}")
                    detail = lookup_and_cache(chosen, qwen_raw)
                    if detail:
                        component_cache = detail
                        _add_component(detail)
                        identify_result = detail['name']
                    else:
                        identify_result = chosen
                    candidates = []
            elif cmd == "cancel":
                manual_input = ""
                candidates = []
                identify_result = ""
                print("【语音】取消")
            elif cmd == "input" or cmd.startswith("input:"):
                model = cmd.split(":", 1)[1] if ":" in cmd else ""
                if model:
                    print(f"【语音】输入型号: {model}")
                    detail = lookup_and_cache(model, qwen_raw)
                    if detail:
                        component_cache = detail
                        _add_component(detail)
                        identify_result = detail['name']
                    else:
                        identify_result = model
                    candidates = []
                else:
                    manual_input = ""
                    candidates = []
                    identify_result = "说型号..."
                    print("【语音】等待说型号")
            elif cmd == "switch_mode":
                # 语音『接线模式』→ 开启接线模式，清空列表，不立即问接线
                _trigger_wiring()
            elif cmd == "wiring":
                if not wiring_mode:
                    _trigger_wiring()  # 还没开接线模式，先开
                n = len(component_list)
                if n == 0:
                    print("【接线】还没有记住任何元件，请先识别")
                else:
                    if wiring_advice and n > last_wiring_count:
                        wiring_advice = ""
                    if not wiring_advice:
                        names = [c.get('name','?')[:10] for c in component_list]
                        msg = f"Tell wiring: I have {' + '.join(names)}. How to connect? Use | per line, max 19 chars."
                        print(f"【语音】询问接线")
                        t = threading.Thread(target=_ask_agent, args=(msg,), daemon=True)
                        t.start()
                        last_wiring_count = n
                        show_wiring = True
                    else:
                        show_wiring = not show_wiring
            elif cmd == "clear":
                _full_reset()

except Exception as main_error:
    print(f"【运行时错误】错误: {main_error}")
    import traceback
    traceback.print_exc()

cap.release()
cv2.destroyAllWindows()
hands.close()
if ser is not None:
    ser.close()
    print("【系统日志】串口已关闭")
if sock is not None:
    sock.close()
    print("【系统日志】UDP 已关闭")
