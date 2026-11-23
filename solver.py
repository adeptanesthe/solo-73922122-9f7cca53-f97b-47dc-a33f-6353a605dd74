"""石刻浅刻线补拍摄影的灯位选择求解核心。

记号
----
* 原照 i 的照明向量为 L_i（三维，任意有限值），像素 p 的实拍灰度为 g_ip；
* 每个像素 p 对应一个反射向量 r_p = (x, y, z)，约束 z >= 0；
* 实拍灰度与反射模型的偏差不超过 2：|L_i . r_p - g_ip| <= 2。

全区共用同一个排除选择 e：不排除任何照片，或排除至多一张被标为不可靠的
原照；标为可靠的照片永远保留。某个 e 下所有像素都可行时，称其为一个
可行解释（可行排除选择）。

备选灯 c 是否合格：在所有可行解释下，全区每个像素的预测灰度 c . r_p
都必须落在 [5, 250]。实现上对每个（像素, 排除选择）求线性规划的最小/
最大值，跨全部可行解释取并集判定。

合格灯的保证值：所有可行解释、所有点对中，底面像素与刻线像素预测灰度
之差（底面减刻线）的最小值。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

# ---- 业务常量 -----------------------------------------------------------
GRAY_TOL = 2.0      # 实拍灰度与点积的最大偏差
GRAY_LOW = 5.0      # 备选灯预测灰度下界
GRAY_HIGH = 250.0   # 备选灯预测灰度上界
BOUND_EPS = 1e-6    # 边界判定的数值容差
POS_EPS = 1e-7      # “保证值为正”的数值阈值


@dataclass
class Scene:
    # images:     shape (k, h, w) 的 uint8 灰度数组
    # lights:     shape (k, 3) 的照明向量
    # candidates: shape (m, 3) 的备选灯向量
    # pairs:      [((line_x, line_y), (base_x, base_y)), ...]
    # reliable:   长度 k 的布尔列表，True 表示该照片可靠、不得排除
    images: np.ndarray
    lights: np.ndarray
    candidates: np.ndarray
    pairs: list
    reliable: list


class _LPSolver:
    """对单个场景的线性规划做惰性构建与缓存。"""

    def __init__(self, scene: Scene):
        self.scene = scene
        self.k, self.h, self.w = scene.images.shape
        self.grays = scene.images.astype(np.float64)
        self._poly_cache: dict = {}
        self._feas_cache: dict = {}
        self._ext_cache: dict = {}

    def _polyhedron(self, pixel: int, exclude):
        """像素 pixel 在排除选择 exclude 下的半空间集合 A r <= b。"""
        key = (pixel, exclude)
        cached = self._poly_cache.get(key)
        if cached is not None:
            return cached

        y, x = divmod(pixel, self.w)
        rows_a, rows_b = [], []
        for i in range(self.k):
            if i == exclude:
                continue
            light = self.scene.lights[i]
            g = self.grays[i, y, x]
            # L.r <= g + 2
            rows_a.append(light)
            rows_b.append(g + GRAY_TOL)
            # -L.r <= 2 - g，即 L.r >= g - 2
            rows_a.append(-light)
            rows_b.append(GRAY_TOL - g)
        # z >= 0  ->  -z <= 0
        rows_a.append(np.array([0.0, 0.0, -1.0]))
        rows_b.append(0.0)

        poly = (np.asarray(rows_a, dtype=np.float64),
                np.asarray(rows_b, dtype=np.float64))
        self._poly_cache[key] = poly
        return poly

    def _solve(self, pixel: int, exclude, obj=None):
        poly_a, poly_b = self._polyhedron(pixel, exclude)
        if obj is None:
            cache_key = ("f", pixel, exclude)
            cached = self._feas_cache.get(cache_key)
            if cached is not None:
                return cached
            obj = np.zeros(3)
        else:
            cache_key = ("e", pixel, exclude, tuple(np.round(obj, 12)))
            cached = self._ext_cache.get(cache_key)
            if cached is not None:
                return cached

        res = linprog(
            obj,
            A_ub=poly_a,
            b_ub=poly_b,
            bounds=[(None, None), (None, None), (None, None)],
            method="highs",
        )
        if obj is None or np.count_nonzero(obj) == 0:
            self._feas_cache[cache_key] = res
        else:
            self._ext_cache[cache_key] = res
        return res

    def feasible(self, pixel: int, exclude) -> bool:
        return bool(self._solve(pixel, exclude).success)

    def extrema(self, pixel: int, exclude, direction: np.ndarray):
        """返回 c.r 在该多面体上的 (最小值, 最大值)。

        多面体无界时端点可能为 -inf / +inf；其它求解失败抛出运行时错误，
        由上层处理，避免静默给出错误的灯位结论。
        """
        res_lo = self._solve(pixel, exclude, direction)
        if res_lo.status == 3:  # LP 无界
            lo = -np.inf
        elif res_lo.success:
            lo = float(res_lo.fun)
        else:
            raise RuntimeError(f"最小值求解失败 status={res_lo.status}")

        res_hi = self._solve(pixel, exclude, -direction)
        if res_hi.status == 3:
            hi = np.inf
        elif res_hi.success:
            hi = float(-res_hi.fun)
        else:
            raise RuntimeError(f"最大值求解失败 status={res_hi.status}")
        return lo, hi


def _feasible_exclusions(solver: _LPSolver):
    """返回所有可行排除选择（None 表示不排除），保持输入顺序。"""
    scene = solver.scene
    options = [None] + [i for i in range(solver.k) if not scene.reliable[i]]
    pixel_count = solver.h * solver.w
    feasible = []
    for exclude in options:
        if all(solver.feasible(p, exclude) for p in range(pixel_count)):
            feasible.append(exclude)
    return feasible


def _judge_light(solver: _LPSolver, direction: np.ndarray,
                 exclusions: list, pair_pixels: list):
    """判定单盏备选灯是否合格，并计算其保证值。"""
    n = solver.h * solver.w
    global_min, global_max = np.inf, -np.inf

    # 1) 合格性：所有可行解释、全区像素的预测灰度都在 [5, 250]
    for exclude in exclusions:
        for pixel in range(n):
            lo, hi = solver.extrema(pixel, exclude, direction)
            if (not np.isfinite(lo) or lo < GRAY_LOW - BOUND_EPS or
                    not np.isfinite(hi) or hi > GRAY_HIGH + BOUND_EPS):
                y, x = divmod(pixel, solver.w)
                reason = (f"像素({x},{y}) 预测灰度可达 "
                          f"{_fmt(lo)} ~ {_fmt(hi)}，超出 [5, 250]")
                return {
                    "qualified": False,
                    "guarantee": None,
                    "global_min": None,
                    "global_max": None,
                    "fail_reason": reason,
                }, None
            global_min = min(global_min, lo)
            global_max = max(global_max, hi)

    # 2) 保证值：所有解释与点对中，最小的（底面 - 刻线）灰度差
    guarantee = np.inf
    for exclude in exclusions:
        for line_p, base_p in pair_pixels:
            base_lo, _ = solver.extrema(base_p, exclude, direction)
            _, line_hi = solver.extrema(line_p, exclude, direction)
            diff = base_lo - line_hi
            if not np.isfinite(diff):
                guarantee = -np.inf
                break
            guarantee = min(guarantee, diff)
    guarantee = float(guarantee)

    return {
        "qualified": True,
        "guarantee": guarantee,
        "global_min": float(global_min),
        "global_max": float(global_max),
        "fail_reason": None,
    }, guarantee


def _fmt(value: float) -> str:
    if np.isneginf(value):
        return "-∞"
    if np.isposinf(value):
        return "+∞"
    return f"{value:.2f}"


def evaluate(scene: Scene) -> dict:
    """主入口：返回可行解释、每盏灯的判定与最终推荐。"""
    solver = _LPSolver(scene)
    exclusions = _feasible_exclusions(solver)

    result = {
        "infeasible": len(exclusions) == 0,
        "feasible_exclusions": exclusions,
        "verdicts": [],
        "recommendation": None,
        "guarantee": None,
    }

    if not exclusions:
        # 无共同解释：不给任何灯位结论，撤下旧推荐由调用方负责。
        for j in range(len(scene.candidates)):
            result["verdicts"].append({
                "index": j,
                "qualified": False,
                "guarantee": None,
                "global_min": None,
                "global_max": None,
                "fail_reason": "原照无共同解释，无法评估",
            })
        return result

    pair_pixels = []
    for (lx, ly), (bx, by) in scene.pairs:
        pair_pixels.append((ly * solver.w + lx, by * solver.w + bx))

    best_index, best_value = None, None
    for j, direction in enumerate(scene.candidates):
        verdict, guarantee = _judge_light(
            solver, direction, exclusions, pair_pixels
        )
        verdict["index"] = j
        result["verdicts"].append(verdict)
        # 仅在严格更大时更新；同分自然保留输入顺序中最先出现的灯。
        if verdict["qualified"] and guarantee > POS_EPS:
            if best_value is None or guarantee > best_value + BOUND_EPS:
                best_index, best_value = j, guarantee

    result["recommendation"] = best_index
    result["guarantee"] = best_value
    return result
