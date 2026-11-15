# 基于 MediaPipe 手势识别的打飞机小游戏

本项目在原 pygame“打飞机”游戏的基础上，把 PaddleX 图像分类手势识别替换为
**MediaPipe Hands**：只对摄像头画面中的绿框区域识别一只手，并在画面上绘制
21 个骨骼点和手部连边。

## 手势映射

| 手势 | 动作 |
| --- | --- |
| 手掌张开 | `up` 向上 |
| 握拳 | `down` 向下 |
| 点赞手势，大拇指朝左 | `left` 向左 |
| 点赞手势，大拇指朝右 | `right` 向右 |
| 其他手势 / 没有手 | `pause` 静止不动 |

手指张开/收缩通过“手指末端节点到手掌心节点的距离”与手掌大小比较判断；
左右方向由大拇指指尖相对大拇指根部的水平位移判断。

## 运行

首次安装（会创建 `.venv` 并安装 MediaPipe 依赖）：

```bat
setup_env.bat
```

之后每次使用：

```bat
:: 1. 检查摄像头（会自动请求管理员权限，UAC 选择“是”）
check_camera.bat

:: 2. 启动游戏（会自动请求管理员权限，UAC 选择“是”）
run_game.bat
```

游戏默认自动扫描 0–5 号摄像头并优先使用摄像头；没有可用摄像头时会回退到
虚拟摄像头 + 方向键。可用 `GAME_NO_ELEVATE=1` 跳过提权（摄像头可能不可用），
用 `GAME_CAMERA=virtual` 强制键盘模式。环境自检：

```bat
.venv\Scripts\python.exe verify_environment.py
```

## 文档

- [修改报告.md](修改报告.md) —— 本次从 PaddleX 切换到 MediaPipe 的修改说明
- [环境配置报告.md](环境配置报告.md) —— 早期 PaddleX 版本的环境配置记录（历史）
- [check_camera.bat](check_camera.bat) / [check_camera.py](check_camera.py) —— 摄像头自检与截图
- [gesture_mediapipe.py](gesture_mediapipe.py) —— MediaPipe 手势识别与骨骼绘制
- [test_gesture_logic.py](test_gesture_logic.py) —— 手势分类逻辑单元测试

## 原项目

- 演示视频：https://www.bilibili.com/video/BV1HV411t7LG/
- 原仓库：https://github.com/Sharpiless/Hand-gesture-recognition-for-game
