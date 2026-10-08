#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2E: 用真实 htdemucs 模型 (CPU) 对 10s 合成音频跑完整分离流程。"""
import os, sys, time, shutil, tempfile
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp(prefix="qa_e2e_", dir=BASE_DIR)
sr = 44100
t = np.linspace(0, 10, 10 * sr, endpoint=False)
mix = (0.4 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
       + 0.05 * np.random.randn(len(t)))
stereo = np.stack([mix, mix], axis=1).astype(np.float32)
wav = os.path.join(tmp, "e2e_input.wav")
sf.write(wav, stereo, sr)

import vocal_separator as vs
from PyQt6.QtWidgets import QApplication
app = QApplication(sys.argv)

done, err = {}, []
t0 = time.time()
worker = vs.AudioSeparationWorker(wav, tmp, model_name="htdemucs",
                                  shifts=1, overlap=0.25, split=True, device="cpu")
from PyQt6.QtCore import Qt
DC = Qt.ConnectionType.DirectConnection
worker.finished_separation.connect(lambda r: done.update(r), DC)
worker.error_occurred.connect(lambda m: err.append(m), DC)
worker.progress_text.connect(lambda s: print(f"  [{time.time()-t0:6.1f}s] {s}"), DC)
worker.start()
worker.wait(600000)  # 最多10分钟

if err:
    print("E2E FAIL:", err[0][:500])
    sys.exit(1)
if not done:
    print("E2E FAIL: 超时未完成")
    sys.exit(1)

print(f"\n耗时 {time.time()-t0:.1f}s, stems={list(done['stems'].keys())}, sr={done['sr']}")
for name, path in done["stems"].items():
    info = sf.info(path)
    data, _ = sf.read(path, always_2d=True)
    print(f"  {name}: {info.duration:.2f}s {info.channels}ch {info.samplerate}Hz "
          f"峰值={np.max(np.abs(data)):.3f} 大小={os.path.getsize(path)//1024}KB")
    assert abs(info.duration - 10.0) < 0.05, f"{name} 时长异常"
    assert info.channels == 2
print("E2E PASS")
shutil.rmtree(tmp, ignore_errors=True)
