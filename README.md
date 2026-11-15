# PaddlePaddle实现手势识别玩转打飞机小游戏！

## 运行代码：

```
python plane_cnn.py
```

## 演示视频：
https://www.bilibili.com/video/BV1HV411t7LG/

![](https://github.com/Sharpiless/Hand-gesture-recognition-for-game/blob/master/demo.png)

## 本机环境配置（Windows / Python 3.12）

详见 [环境配置报告.md](环境配置报告.md)。

首次使用：

```bat
:: 1. 检查摄像头（会自动请求管理员权限，UAC 选择“是”）
check_camera.bat

:: 2. 启动游戏（会自动请求管理员权限，UAC 选择“是”）
run_game.bat
```

游戏默认自动扫描 0–5 号摄像头并优先使用摄像头；没有可用摄像头时会回退到
虚拟摄像头 + 方向键。若只想用键盘模式，可设置 `GAME_NO_ELEVATE=1` 和
`GAME_CAMERA=virtual`，这样不会请求管理员权限。若不希望 UAC 提权，可先确认
“设置 → 隐私和安全性 → 相机 → 让桌面应用访问你的摄像头”已打开，再设置
`GAME_NO_ELEVATE=1`。环境自检：

```bat
.venv\Scripts\python.exe verify_environment.py
```
