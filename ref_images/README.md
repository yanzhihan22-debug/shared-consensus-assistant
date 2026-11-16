# ref_images

这里放你自己的元件参考照片，用于 OpenCLIP 本地库。

目录约定：

```
ref_images/
├─ SG90/
│  ├─ 1.jpg
│  └─ 2.jpg
├─ ST7789/
│  └─ 1.jpg
└─ ESP32-S3-CAM/
   └─ 1.jpg
```

添加后：

```bash
python build_db.py
# 选择“同步照片 (扫描 ref_images 文件夹)”
```

说明：

- 照片不提交到 GitHub（见 `.gitignore`），避免个人桌面/手部隐私；
- 每个型号 1–3 张、不同角度/光照即可；
- 参考照片越接近实际拍摄环境，CLIP 快速通道越准。
