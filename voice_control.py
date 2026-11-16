"""
PC 麦克风语音控制 — 科大讯飞语音听写 WebSocket API
独立线程运行，通过 queue 向主程序发命令
"""

import os
from dotenv import load_dotenv
load_dotenv()
import threading
import queue
import numpy as np
import time
import json
import hashlib
import hmac
import base64
from datetime import datetime
from urllib.parse import urlencode, quote_plus

# ---- 讯飞配置（从环境变量读取，勿提交真实凭据）----
XF_APPID = os.getenv("XF_APPID", "")
XF_API_SECRET = os.getenv("XF_API_SECRET", "")
XF_API_KEY = os.getenv("XF_API_KEY", "")

SAMPLE_RATE = 16000
CHUNK_SEC = 0.3
SILENCE_SEC = 0.8
MAX_SPEECH_SEC = 3.0
VAD_THRESHOLD = 0.004

COMMANDS = {
    "unlock":   ["退出", "推出", "解冻", "解锁", "解除", "接触"],
    "lock":     ["锁定", "锁住", "冻结"],
    "identify": ["识别", "看看", "是什么"],
    "confirm":  ["确认", "确定", "是", "对的"],
    "select_1": ["第一个", "选1", "1号"],
    "select_2": ["第二个", "选2", "2号"],
    "select_3": ["第三个", "选3", "3号"],
    "wiring":   ["接线", "连接", "怎么接"],
    "switch_mode": ["切换模式", "换模式", "接线模式", "切模式"],
    "clear":    ["清空", "重置", "重新开始"],
    "cancel":   ["取消", "算了", "不要"],
    "input":    ["输入", "手动", "型号"],
}

def _build_xf_url():
    """生成讯飞 WebSocket 鉴权 URL"""
    host = "iat-api.xfyun.cn"
    path = "/v2/iat"
    date = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
    sig = hmac.new(XF_API_SECRET.encode(), signature_origin.encode(), hashlib.sha256).digest()
    sig_b64 = base64.b64encode(sig).decode()
    auth_origin = (f'api_key="{XF_API_KEY}", algorithm="hmac-sha256", '
                   f'headers="host date request-line", signature="{sig_b64}"')
    auth_b64 = base64.b64encode(auth_origin.encode()).decode()
    params = urlencode({"authorization": auth_b64, "date": date, "host": host})
    return f"wss://{host}{path}?{params}"

def _xf_req_json(status, audio_b64):
    """构造讯飞请求 JSON"""
    return json.dumps({
        "common": {"app_id": XF_APPID},
        "business": {"language": "zh_cn", "domain": "iat", "accent": "mandarin"},
        "data": {"status": status, "format": "audio/L16;rate=16000", "encoding": "raw", "audio": audio_b64}
    })

def transcribe_xf(audio_bytes):
    """调用讯飞 WebSocket 转写一段音频（≤60秒）"""
    import websocket
    url = _build_xf_url()
    result = [""]
    lock = threading.Lock()
    done = threading.Event()

    def on_msg(ws, msg):
        try:
            d = json.loads(msg)
            if d.get("code") != 0:
                print(f"【语音】讯飞错误: {d.get('message','')}")
                done.set(); return
            if d.get("data", {}).get("result", {}).get("ws"):
                parts = []
                for w in d["data"]["result"]["ws"]:
                    for cw in w.get("cw", []):
                        parts.append(cw.get("w", ""))
                txt = "".join(parts)
                with lock:
                    result[0] += txt
            if d.get("data", {}).get("status") == 2:
                done.set()
        except Exception as e:
            print(f"【语音】解析错误: {e}")

    def on_err(ws, e):
        print(f"【语音】WS错误: {e}")
        done.set()

    def on_close(ws, *a):
        done.set()

    ws = websocket.WebSocketApp(url, on_message=on_msg, on_error=on_err,
                                on_close=on_close, on_open=lambda ws: None)
    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    # 发送音频（<=60秒）
    chunk_size = 8000
    total = len(audio_bytes)
    sent = 0
    while sent < total:
        chunk = audio_bytes[sent:sent+chunk_size]
        status = 0 if sent == 0 else (2 if sent + chunk_size >= total else 1)
        ws.send(_xf_req_json(status, base64.b64encode(chunk).decode()))
        sent += chunk_size
        time.sleep(0.04)

    done.wait(timeout=10)
    ws.close()
    with lock:
        return result[0].strip()

def match_command(text):
    """模糊匹配命令关键词；无匹配时若像型号（字母+数字）则作为型号输入"""
    import re
    for cmd, keywords in COMMANDS.items():
        for kw in keywords:
            if kw in text:
                if cmd == "input":
                    m = re.search(r'(?:输入|手动|型号)[\s:：]*([A-Za-z0-9][A-Za-z0-9\-_.]*)', text)
                    if m:
                        return f"input:{m.group(1)}"
                    return "input"
                return cmd
    m = re.search(r'[A-Za-z][A-Za-z0-9\-_.]*\d[A-Za-z0-9\-_.]*', text)
    if m:
        model = m.group(0)
        if len(model) >= 2:
            return f"input:{model}"
    return None

def run_voice_control(cmd_queue: queue.Queue):
    """语音控制主线程"""
    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                    input=True, frames_per_buffer=int(SAMPLE_RATE * CHUNK_SEC))
    print("【语音】麦克风已打开，等待语音指令（讯飞识别）...")

    chunk_samples = int(SAMPLE_RATE * CHUNK_SEC)
    buf = np.array([], dtype=np.int16)
    speech_buf = np.array([], dtype=np.int16)
    speaking = False
    silence_frames = 0

    def rms(data):
        if len(data) == 0: return 0
        return float(np.sqrt(np.mean(np.square(data.astype(np.float32) / 32768.0))))

    while True:
        try:
            raw = stream.read(chunk_samples, exception_on_overflow=False)
        except Exception as e:
            print(f"【语音】读取错误: {e}")
            continue
        samples = np.frombuffer(raw, dtype=np.int16)
        buf = np.append(buf, samples)
        while len(buf) >= chunk_samples:
            chunk = buf[:chunk_samples].copy()
            buf = buf[chunk_samples:]
            has_voice = rms(chunk) > VAD_THRESHOLD
            if has_voice and not speaking:
                speaking = True
                speech_buf = chunk.copy()
                silence_frames = 0
            elif speaking:
                silence_frames = 0 if has_voice else silence_frames + 1
                speech_buf = np.append(speech_buf, chunk)
                if (silence_frames * CHUNK_SEC) >= SILENCE_SEC or len(speech_buf) >= SAMPLE_RATE * MAX_SPEECH_SEC:
                    try:
                        text = transcribe_xf(speech_buf.tobytes())
                    except Exception as e:
                        print(f"【语音】识别失败: {e}")
                        text = ""
                    if text:
                        print(f"【语音】> {text}")
                        cmd = match_command(text)
                        if cmd:
                            print(f"【语音】→ 命令: {cmd}")
                            cmd_queue.put(cmd)
                        else:
                            print(f"【语音】无匹配命令")
                    speaking = False
                    speech_buf = np.array([], dtype=np.int16)
                    silence_frames = 0

    stream.stop_stream(); stream.close(); p.terminate()
