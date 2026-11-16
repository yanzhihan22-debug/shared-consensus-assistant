# datasheets（可选）

把元件 PDF 数据手册放在这里，`component_kb.py` 会按文件名/关键词匹配并抽取引脚。

- 文件名建议包含型号，如 `ST7789V_pinout.pdf`、`SG90.pdf`；
- 没有 PDF 也能跑：会走 `components_db.json` 缓存或 DeepSeek 通用知识兜底；
- PDF 不提交到仓库（见 `.gitignore`），避免版权/体积问题。
