"""淘宝登录：打开淘宝首页，等用户扫码登录，检测到登录态后保存 storage_state"""
import os
import sys
from playwright.sync_api import sync_playwright

STATE_FILE = os.path.join(os.path.dirname(__file__), "taobao_state.json")

def is_logged_in(page):
    """通过页面是否出现登录后标记判断"""
    try:
        return page.evaluate("""() => {
            const t = document.body.innerText || '';
            // 淘宝登录后通常出现 我的淘宝/退出 或昵称，且不再有"亲，请登录"
            if (/亲，请登录/.test(t)) return false;
            if (/退出/.test(t)) return true;
            // 无 head 标记时看是否有 购物车数字/账户相关
            return /我的淘宝|免费注册/.test(t) === false;
        }""")
    except Exception:
        return False

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=[
            "--disable-blink-features=AutomationControlled",
            "--window-size=1280,900",
        ])
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        )
        page = context.new_page()
        page.goto("https://www.taobao.com", wait_until="domcontentloaded", timeout=60000)
        print("请在浏览器里完成登录（扫码或账号）。检测到登录态后自动保存。")
        # 若已带旧 cookie 就直接检测
        if is_logged_in(page):
            print("已检测到登录态（使用旧会话）")
        else:
            # 跳转登录页引导用户
            try:
                page.goto("https://login.taobao.com/member/login.jhtml", wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(2000)
                # 优先点"密码登录"（可选），默认等待扫码
            except Exception:
                pass
            print("等待登录...（上限 5 分钟）")
            ok = False
            try:
                page.wait_for_function(
                    """() => {
                        const t = document.body.innerText || '';
                        return /亲，请登录/.test(t) === false && (/退出/.test(t) || /我的淘宝/.test(t));
                    }""",
                    timeout=300000,
                )
                ok = True
            except Exception:
                # 兜底再查一次
                ok = is_logged_in(page)
            if not ok:
                print("[警告] 未能确认登录态，仍保存当前 cookies")
        page.wait_for_timeout(2000)
        context.storage_state(path=STATE_FILE)
        cookies = context.cookies()
        names = [c["name"] for c in cookies]
        print(f"已保存 {len(cookies)} 个 cookie -> {STATE_FILE}")
        print("含登录标记 _uc_lgba:", "_uc_lgba" in names)
        print("含 cookie2:", "cookie2" in names)
        browser.close()

if __name__ == "__main__":
    main()
