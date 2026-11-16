"""测试外接摄像头的实际帧率与支持分辨率"""
import cv2, time

def bench(cap, w, h, fps_set, dur=3.0):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if fps_set:
        cap.set(cv2.CAP_PROP_FPS, fps_set)
    for _ in range(10):
        cap.grab()
    t0 = time.time(); n = 0
    while time.time() - t0 < dur:
        ret, _ = cap.read()
        if not ret:
            break
        n += 1
    dt = time.time() - t0
    act_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    act_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    act_fps = cap.get(cv2.CAP_PROP_FPS)
    return n / dt, act_w, act_h, act_fps

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无摄像头"); return
    print("当前默认: %.0fx%.0f @ %.0ffps" % (cap.get(3), cap.get(4), cap.get(5)))
    print("\n--- 各组合实测（读取帧率，3秒平均）---")
    tests = [(640,480,30), (640,480,60), (800,600,30), (1280,720,30), (1280,720,60), (1920,1080,30)]
    for w,h,f in tests:
        real, aw, ah, af = bench(cap, w, h, f)
        print(f"  设 {w}x{h}@{f}fps -> 实测 {real:5.1f}fps (返回 {aw:.0f}x{ah:.0f} @ {af:.0f}fps)")
    cap.release()

if __name__ == "__main__":
    main()
