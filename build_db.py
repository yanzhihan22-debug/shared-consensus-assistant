"""
手动建库工具 — 输入型号+引脚+图片 → 生成 MANUAL_DB.json
用法: python build_db.py
"""
import json
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "MANUAL_DB.json")
REF_DIR = os.path.join(BASE_DIR, "ref_images")

def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def ask(question, default=""):
    val = input(question + (" [%s]: " % default if default else ": ")).strip()
    return val or default

def collect_photos(entry_name):
    """输入照片路径，拷贝到 ref_images/<name>/"""
    os.makedirs(REF_DIR, exist_ok=True)
    folder = os.path.join(REF_DIR, entry_name)
    os.makedirs(folder, exist_ok=True)
    print(f"\n照片会拷贝到: {folder}")
    print("输入照片路径（可拖动文件到窗口），一行一个，空行结束。没有直接回车。")
    paths = []
    while True:
        p = input("  照片: ").strip().strip('"').strip("'")
        if not p:
            break
        if os.path.isfile(p):
            fname = f"{len(paths)+1}{os.path.splitext(p)[1].lower()}"
            dst = os.path.join(folder, fname)
            shutil.copy2(p, dst)
            paths.append(dst)
            print(f"    ✓ 已保存 {dst}")
        else:
            print(f"    ✗ 文件不存在: {p}")
    return [os.path.relpath(p, BASE_DIR) for p in paths]

def add_entry(db):
    print("\n=== 添加元件 ===")
    name = ask("型号 (如 LM358, SG90, ST7789)", "")
    if not name:
        print("型号不能为空，取消。")
        return
    key = name.lower().replace(" ", "_")
    if key in db:
        print(f"已存在 {name}，如需修改请用编辑功能。")
        return

    print("引脚信息（一行一个，格式: 引脚名 描述，空行结束）")
    print("  示例: VCC 电源正极 / OUT 信号输出")
    pins = []
    while True:
        line = input("  引脚: ").strip()
        if not line:
            break
        parts = line.split(" ", 1)
        pin = parts[0]
        desc = parts[1] if len(parts) > 1 else ""
        pins.append({"pin": pin, "desc": desc})

    photos = collect_photos(name)

    entry = {
        "name": name,
        "pins": pins,
        "photos": photos,
    }
    db[key] = entry
    save_db(db)
    print(f"\n✓ 已保存 {name}")

def show_db(db):
    if not db:
        print("库是空的。")
        return
    print("\n当前数据库:")
    for k, e in db.items():
        print(f"  {e['name']} | 引脚: {len(e.get('pins', []))} 个 | 照片: {len(e.get('photos', []))} 张")

def sync_photos(db):
    """扫描 ref_images/<型号>/ 下所有照片，自动同步进库"""
    if not os.path.isdir(REF_DIR):
        print("没有 ref_images 文件夹。")
        return
    changed = 0
    for folder in os.listdir(REF_DIR):
        fpath = os.path.join(REF_DIR, folder)
        if not os.path.isdir(fpath):
            continue
        key = folder.lower().replace(" ", "_")
        files = sorted(f for f in os.listdir(fpath)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
        photos = [os.path.relpath(os.path.join(fpath, f), BASE_DIR).replace("\\", "/") for f in files]
        if not photos:
            continue
        if key not in db:
            print(f"新元件: {folder} ({len(photos)} 张照片)")
            db[key] = {"name": folder, "pins": [], "photos": photos}
            changed += 1
        else:
            old = db[key].get("photos", [])
            new_added = [p for p in photos if p not in old]
            removed = [p for p in old if p not in photos]
            db[key]["photos"] = photos
            if new_added or removed:
                print(f"更新 {folder}: +{len(new_added)} 张 / -{len(removed)} 张")
                changed += 1
    if changed:
        save_db(db)
        print(f"\n已同步 {changed} 个条目。")
    else:
        print("没有变化。")

def main():
    db = load_db()
    while True:
        print("\n========== 手动建库 ==========")
        print("1. 添加元件")
        print("2. 查看数据库")
        print("3. 同步照片 (扫描 ref_images 文件夹)")
        print("4. 退出")
        c = input("选择: ").strip()
        if c == "1":
            add_entry(db)
        elif c == "2":
            show_db(db)
        elif c == "3":
            sync_photos(db)
        elif c == "4":
            print("退出。")
            break
        else:
            print("无效输入。")

if __name__ == "__main__":
    main()
