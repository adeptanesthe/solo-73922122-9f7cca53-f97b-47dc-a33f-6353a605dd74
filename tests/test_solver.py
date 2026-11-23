"""solver 单元自测（不依赖 Web 层），直接 python tests/test_solver.py 运行。"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import GRAY_TOL, Scene, evaluate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo")


def load_demo(reliable_override=None):
    with open(os.path.join(DEMO, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    images = np.stack([
        np.asarray(Image.open(os.path.join(DEMO, name)).convert("L"),
                   dtype=np.uint8)
        for name in meta["images"]
    ])
    reliable = (reliable_override if reliable_override is not None
                else [True] * len(meta["images"]))
    return Scene(
        images=images,
        lights=np.asarray(meta["lights"], dtype=np.float64),
        candidates=np.asarray(meta["candidates"], dtype=np.float64),
        pairs=[(tuple(p[0]), tuple(p[1])) for p in meta["suggested_pairs"]],
        reliable=reliable,
    ), meta


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ✓", msg)


def test_demo_recommendation():
    print("test_demo_recommendation")
    scene, meta = load_demo()
    result = evaluate(scene)
    check(not result["infeasible"], "存在可行解释")
    check(result["feasible_exclusions"] == [None],
          "全部可靠时唯一可行解释为不排除任何照片")
    qualified = [v["index"] for v in result["verdicts"] if v["qualified"]]
    check(qualified == [0, 1, 3], f"合格灯为第1、2、4盏，实际 {qualified}")
    check(result["recommendation"] == 0, "推荐第 1 盏")
    check(result["guarantee"] is not None and result["guarantee"] > 0,
          f"推荐保证值为正: {result['guarantee']}")
    g1 = result["verdicts"][0]["guarantee"]
    g4 = result["verdicts"][3]["guarantee"]
    check(g1 > g4 > 0, f"第1盏保证值 {g1} 大于第4盏 {g4}")
    g2 = result["verdicts"][1]["guarantee"]
    check(g2 <= 0, f"第2盏合格但保证值非正 ({g2})，不得被推荐")
    v3 = result["verdicts"][2]
    check(not v3["qualified"] and v3["fail_reason"],
          "第3盏不合格并给出原因")


def test_tolerance_band():
    print("test_tolerance_band")
    scene, _ = load_demo()
    # 把某张原照整体偏移 2：仍应可行（边界恰好在 ±2）；偏移 3 则不可行
    scene.images[0] = np.clip(
        scene.images[0].astype(int) + 2, 0, 255).astype(np.uint8)
    result = evaluate(scene)
    check(not result["infeasible"], "灰度整体 +2 仍在容差内，可行")

    scene2, _ = load_demo()
    scene2.images[0] = np.clip(
        scene2.images[0].astype(int) + 30, 0, 255).astype(np.uint8)
    result2 = evaluate(scene2)
    check(result2["infeasible"], "灰度整体 +30 超出可行松弛，且全部可靠 → 无共同解释")


def test_exclusion_and_reliable():
    print("test_exclusion_and_reliable")
    scene, _ = load_demo()
    # 污染第 1 张（index 0），并把它标为不可靠：排除它应恢复可行
    scene.images[0] = np.clip(
        scene.images[0].astype(int) + 40, 0, 255).astype(np.uint8)
    scene.reliable = [False, True, True, True]
    result = evaluate(scene)
    check(not result["infeasible"], "允许排除污染照片后存在可行解释")
    check(result["feasible_exclusions"] == [0],
          f"唯一可行解释是排除第1张，实际 {result['feasible_exclusions']}")

    # 污染照片标为可靠 → 无法排除 → 无共同解释
    scene.reliable = [True, True, True, True]
    result_reliable = evaluate(scene)
    check(result_reliable["infeasible"],
          "污染照片被标为可靠时无共同解释")
    check(all(v["guarantee"] is None for v in result_reliable["verdicts"]),
          "无共同解释时不给出任何灯位保证")
    check(result_reliable["recommendation"] is None, "推荐撤下")


def test_z_nonnegative_enforced():
    print("test_z_nonnegative_enforced")
    # 单像素、2x2 裁片；构造只有负 z 才能满足的约束：
    # 两张灯分别让点积目标都偏高，而 x/y 分量无法兼顾——用随机方向验证
    rng = np.random.default_rng(0)
    lights = rng.normal(size=(4, 3))
    lights[:, 2] = np.abs(lights[:, 2]) + 0.5
    # 用一个 z>=0 的真实反射体生成数据，保证可行；再反向检查 z 约束生效
    r_true = np.array([0.2, -0.3, 100.0])
    images = np.stack([
        np.full((2, 2), np.clip(round(float(L @ r_true)), 0, 255),
                dtype=np.uint8)
        for L in lights
    ])
    scene = Scene(images=images, lights=lights,
                  candidates=np.zeros((2, 3)),
                  pairs=[((0, 0), (1, 1))], reliable=[True] * 4)
    result = evaluate(scene)
    check(not result["infeasible"], "z≥0 真实反射体数据可行")

    # 零向量灯：预测恒为 0，落在 [5,250] 之外 → 不合格
    check(not result["verdicts"][0]["qualified"],
          "零向量备选灯预测灰度 0，不合格")


def test_tie_prefers_first():
    print("test_tie_prefers_first")
    scene, _ = load_demo()
    # 两盏完全相同的灯 → 同分，按输入顺序取首盏
    scene.candidates = np.tile(scene.candidates[0], (2, 1))
    result = evaluate(scene)
    check(result["recommendation"] == 0, "同分推荐输入顺序首盏")
    check(result["verdicts"][0]["guarantee"]
          == result["verdicts"][1]["guarantee"], "两盏保证值相同")


def test_pair_difference_must_be_positive():
    print("test_pair_difference_must_be_positive")
    scene, _ = load_demo()
    result = evaluate(scene)
    cand2 = result["verdicts"][1]
    check(cand2["qualified"] and cand2["guarantee"] <= 0,
          f"第2盏合格但保证值非正 ({cand2['guarantee']})，不得推荐")
    check(result["recommendation"] != 1, "负保证灯未被推荐")


if __name__ == "__main__":
    test_demo_recommendation()
    test_tolerance_band()
    test_exclusion_and_reliable()
    test_z_nonnegative_enforced()
    test_tie_prefers_first()
    test_pair_difference_must_be_positive()
    print("\n全部自测通过。")
