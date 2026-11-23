"""生成演示用灰度裁片（4x4，4 张原照）与备选灯元数据。

模型与求解器一致：灰度 g = L · r（取整），其中 r=(x,y,z)，z≥0。
裁片包含两类像素：
* 底面像素：反射向量接近 (0, 0, 240)；
* 刻线像素：带有横向/纵向倾斜，点积随灯位变化更明显。
生成结果全部落在 [5, 250]，避免 PNG 截断。
"""
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

WIDTH = HEIGHT = 4

# 每像素反射向量 (x, y, z)，z≥0
REFLECTORS = {
    "base": np.array([0.0, 0.0, 240.0]),
    "line_x": np.array([120.0, 0.0, 180.0]),
    "line_y": np.array([0.0, 120.0, 180.0]),
}

# 4 张原照的照明向量
LIGHTS = [
    [0.25, 0.10, 0.85],
    [-0.20, 0.30, 0.90],
    [0.15, -0.25, 0.80],
    [-0.10, -0.15, 0.95],
]

# 像素布局：B=底面，X=x 向刻线，Y=y 向刻线
LAYOUT = [
    ["B", "B", "B", "B"],
    ["B", "X", "Y", "B"],
    ["B", "B", "B", "B"],
    ["B", "X", "Y", "B"],
]

# 备选灯（数值经 LP 可行域实测标定）：
# 1) 合格、正保证，且保证值最大（预期被推荐）
# 2) 合格但保证值为负（刻线极值可反超底面，不推荐）
# 3) 不合格：部分像素预测灰度越过 [5,250]
# 4) 合格、正保证但保证值低于第 1 盏（次优，用于验证排序）
CANDIDATES = [
    [0.00, 0.00, 0.90],
    [0.70, 0.00, 0.70],
    [1.50, 0.00, 0.90],
    [0.10, 0.10, 0.80],
]


def main():
    reflector_grid = np.zeros((HEIGHT, WIDTH, 3))
    for y, row in enumerate(LAYOUT):
        for x, kind in enumerate(row):
            key = {"B": "base", "X": "line_x", "Y": "line_y"}[kind]
            reflector_grid[y, x] = REFLECTORS[key]

    images = []
    for i, light in enumerate(LIGHTS):
        light = np.asarray(light)
        raw = reflector_grid @ light
        gray = np.rint(raw).astype(np.uint8)
        lo, hi = int(gray.min()), int(gray.max())
        assert 5 <= lo and hi <= 250, f"演示图 {i} 灰度越界: {lo}..{hi}"
        images.append(gray)
        Image.fromarray(gray, mode="L").save(
            os.path.join(HERE, f"demo_{i}.png"))

    meta = {
        "size": [WIDTH, HEIGHT],
        "images": [f"demo_{i}.png" for i in range(len(LIGHTS))],
        "lights": LIGHTS,
        "candidates": CANDIDATES,
        # 与 LAYOUT 对应的建议点对（刻线, 底面）
        "suggested_pairs": [
            [[1, 1], [0, 0]],
            [[2, 1], [3, 1]],
        ],
        "layout": LAYOUT,
    }
    with open(os.path.join(HERE, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    # 打印理论范围，方便人工核对
    print("候选灯在各像素类型上的理论灰度 L·r：")
    for j, cand in enumerate(CANDIDATES):
        cand = np.asarray(cand)
        vals = {name: float(r @ cand) for name, r in REFLECTORS.items()}
        print(f"  灯{j + 1} {cand.tolist()}: " +
              ", ".join(f"{k}={v:.1f}" for k, v in vals.items()))


if __name__ == "__main__":
    main()
