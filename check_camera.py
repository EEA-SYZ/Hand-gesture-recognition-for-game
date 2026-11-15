# -*- coding: utf-8 -*-
"""Camera self-check for the hand-gesture game.

Run it from a normal desktop session (double-click check_camera.bat) so the
process gets the regular user token.  It lists the Windows camera devices,
tries OpenCV indices 0-5 with DirectShow / Media Foundation / AUTO backends,
and saves the first successful frame as camera_snapshot.png.

Exit codes:
    0  camera opened and a frame was saved
    2  Windows sees camera device(s) but OpenCV could not open them
    3  no camera device was found
"""

from __future__ import absolute_import

import os
import re
import subprocess
import sys
import time

import cv2
import ctypes

try:  # Keep the console readable; we print our own diagnostics.
    cv2.setLogLevel(0)
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

SNAPSHOT = "camera_snapshot.png"


def list_pnp_cameras():
    """Return camera devices reported by pnputil (best effort)."""
    try:
        proc = subprocess.run(
            ["pnputil", "/enum-devices", "/class", "Camera"],
            capture_output=True, text=True, errors="replace", timeout=20)
    except Exception as exc:
        return [], "pnputil failed: {}".format(exc)
    out = proc.stdout or ""
    devices = []
    for block in re.split(r"(?=Instance ID:)", out):
        match = re.search(r"Instance ID:\s*(\S+)", block)
        if not match:
            continue
        name = re.search(r"Device Description:\s*(.+)", block)
        status = re.search(r"Status:\s*(\S+)", block)
        devices.append({
            "id": match.group(1).strip(),
            "name": name.group(1).strip() if name else "(unknown)",
            "status": status.group(1).strip() if status else "(unknown)",
        })
    if not devices:
        # Localized pnputil output may not contain the English labels above.
        ids = set(re.findall(
            r"USB\\VID_[0-9A-Fa-f]{4}&PID_[0-9A-Fa-f]{4}[^\s]*", out))
        devices = [{"id": item, "name": "(camera device)",
                    "status": "?"} for item in ids]
    return devices, None


def try_opencv(max_index=6):
    """Try each index/backend; return (success, snapshot_or_None)."""
    for index in range(max_index):
        results = []
        for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
            backend = getattr(cv2, name, None)
            if backend is None:
                continue
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                results.append(name + ":no")
                continue
            results.append(name + ":open")
            # Give UVC cameras a moment to produce their first frame.
            for _ in range(15):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    print("  index {} / {} -> {}x{}, 已保存 {}".format(
                        index, name, frame.shape[1], frame.shape[0], SNAPSHOT))
                    cv2.imwrite(os.path.join(ROOT, SNAPSHOT), frame)
                    cap.release()
                    return True, frame
                time.sleep(0.05)
            cap.release()
            results[-1] = name + ":no-frame"
        print("  index {}: {}".format(index, ", ".join(results)))
    return False, None


def main():
    print("=" * 66)
    print("摄像头自检 / Camera check")
    print("=" * 66)
    print("Python:", sys.version.split()[0])
    print("OpenCV:", cv2.__version__)
    try:
        is_admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        is_admin = False
    print("管理员:", "是" if is_admin else "否（相机可能被系统拒绝）")
    print()

    print("[1/2] Windows 摄像头设备")
    devices, error = list_pnp_cameras()
    if error:
        print("  !", error)
    elif not devices:
        print("  Windows 未报告 Camera 类设备。")
    else:
        for dev in devices:
            print("  - {} | {} | {}".format(
                dev["name"], dev["status"], dev["id"]))
    print()

    print("[2/2] OpenCV 打开测试（索引 0-5，DirectShow / MSMF / AUTO）")
    ok, frame = try_opencv()
    print()

    if ok:
        print("结果：摄像头可用。已保存 camera_snapshot.png。")
        print("现在可以双击 run_game.bat，游戏会自动使用摄像头。")
        return 0

    print("结果：OpenCV 未能打开摄像头。")
    if devices:
        print("Windows 能看到摄像头设备，说明硬件和驱动存在，问题在访问权限或占用：")
    else:
        print("Windows 也没有报告摄像头设备，请检查硬件连接。")
    print("  1. 关闭可能占用摄像头的程序：微信、腾讯会议、浏览器、相机 App。")
    print("  2. Windows 设置 → 隐私和安全性 → 相机：")
    print("     - 打开“摄像头访问”")
    print("     - 打开“让桌面应用访问你的摄像头”")
    print("  3. 检查笔记本的摄像头隐私开关/物理快门（常见于 Fn 功能键）。")
    print("  4. 本机已验证：右键“以管理员身份运行” check_camera.bat 可以正常出图。")
    print("     双击时脚本会自动请求管理员权限（UAC）。")
    print("  5. 仍失败时，在设备管理器中禁用再启用摄像头，或更新驱动。")
    return 2 if devices else 3


if __name__ == "__main__":
    sys.exit(main())
