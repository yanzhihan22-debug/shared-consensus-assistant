# 演示视频

仓库不直接提交 MP4（避免仓库膨胀）。推荐两种方式：

## 方式 A：GitHub Releases（推荐）

1. 打开仓库右侧 **Releases** → **Draft a new release**
2. Tag 填 `v1.0-demo`，标题填 `Demo videos`
3. 把下面三个文件拖到 **Attach binaries**：
   - `识别模式.mp4`（约 8.1 MB）
   - `接线模式.mp4`（约 7.5 MB）
   - `淘宝链接.mp4`（约 0.8 MB）
4. Publish release
5. 复制三个 asset 的下载链接，替换 README 顶部“演示视频”处

示例：

```markdown
> 🎬 演示视频：
> [识别模式](https://github.com/yanzhihan22-debug/shared-consensus-assistant/releases/download/v1.0-demo/recognition.mp4) ·
> [接线模式](https://github.com/yanzhihan22-debug/shared-consensus-assistant/releases/download/v1.0-demo/wiring.mp4) ·
> [淘宝链接](https://github.com/yanzhihan22-debug/shared-consensus-assistant/releases/download/v1.0-demo/taobao.mp4)
```

## 方式 B：Bilibili / YouTube

上传后把链接放到 README 顶部即可；GitHub 不会托管视频，但 README 可以外链。

## 建议

- 视频前 3 秒放“最终效果”，再放调试过程；
- 加中文字幕/关键步骤标注；
- 如果要在 README 内嵌预览，截 5–8 秒做 GIF（< 10 MB），放 `docs/images/demo.gif`。
