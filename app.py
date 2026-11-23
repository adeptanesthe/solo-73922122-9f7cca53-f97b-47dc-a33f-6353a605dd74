"""石刻浅刻线补拍灯位选择 —— Flask Web 应用。

流程：
1. 上传 4~8 张已对齐的灰度 PNG 裁片（同尺寸、每边至多 16 像素），
   填写每张原照的三维照明向量与 2~8 个备选灯向量；
2. 在放大裁片上点选 1~4 对（刻线像素, 底面像素），两点必须不同；
3. 后端用 Pillow 读图、SciPy(HiGHS) 求线性规划，给出合格灯与推荐；
4. 更改“可靠”标记后重新求解，推荐随结果更新或撤下。
"""
from __future__ import annotations

import io
import json
import math
import os
import time
import uuid

import numpy as np
from flask import (Flask, jsonify, render_template, request,
                   send_file, session)
from PIL import Image

from solver import GRAY_HIGH, GRAY_LOW, GRAY_TOL, Scene, evaluate

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# 单用户工具：案例保存在进程内存中，键为随机案例 id。
CASES: dict = {}

MIN_IMAGES, MAX_IMAGES = 4, 8
MIN_CANDIDATES, MAX_CANDIDATES = 2, 8
MIN_PAIRS, MAX_PAIRS = 1, 4
MAX_SIDE = 16


# --------------------------------------------------------------------------
# 校验与解析
# --------------------------------------------------------------------------
class ValidationError(Exception):
    pass


def _parse_vec3(values, label):
    if len(values) != 3:
        raise ValidationError(f"{label} 需要 3 个分量")
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValidationError(f"{label} 含非法数字: {v!r}")
        if not math.isfinite(f):
            raise ValidationError(f"{label} 含非有限数值")
        out.append(f)
    return out


def _load_grayscale_png(file_storage, label):
    """用 Pillow 读取，强制要求 PNG 格式、灰度模式。"""
    if file_storage is None or not file_storage.filename:
        raise ValidationError(f"{label} 缺少文件")
    if not file_storage.filename.lower().endswith(".png"):
        raise ValidationError(f"{label} 必须是 PNG 文件")
    try:
        img = Image.open(io.BytesIO(file_storage.read()))
        img.load()
    except Exception as exc:  # Pillow 会抛各类 OSError
        raise ValidationError(f"{label} 无法作为图片读取: {exc}")
    if img.format != "PNG":
        raise ValidationError(f"{label} 不是 PNG 格式")
    if img.mode != "L":
        raise ValidationError(
            f"{label} 必须是灰度图（模式 L），当前为 {img.mode}")
    if img.width < 1 or img.height < 1:
        raise ValidationError(f"{label} 尺寸非法")
    if img.width > MAX_SIDE or img.height > MAX_SIDE:
        raise ValidationError(
            f"{label} 每边至多 {MAX_SIDE} 像素，"
            f"当前 {img.width}x{img.height}")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return np.asarray(img, dtype=np.uint8), buf.getvalue(), img.size


def _validate_pairs(pairs, width, height):
    if not isinstance(pairs, list) or not (MIN_PAIRS <= len(pairs) <= MAX_PAIRS):
        raise ValidationError(
            f"点对数量须在 {MIN_PAIRS}~{MAX_PAIRS} 之间")
    cleaned = []
    for idx, pair in enumerate(pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValidationError(f"第 {idx + 1} 对点结构非法")
        points = []
        for point in pair:
            if not isinstance(point, list) or len(point) != 2:
                raise ValidationError(f"第 {idx + 1} 对点坐标非法")
            x, y = point
            if (not isinstance(x, int) or not isinstance(y, int)
                    or not (0 <= x < width) or not (0 <= y < height)):
                raise ValidationError(
                    f"第 {idx + 1} 对点坐标 ({x},{y}) 超出裁片范围")
            points.append((x, y))
        if points[0] == points[1]:
            raise ValidationError(f"第 {idx + 1} 对点的两个像素必须不同")
        cleaned.append((points[0], points[1]))
    return cleaned


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        min_images=MIN_IMAGES, max_images=MAX_IMAGES,
        min_candidates=MIN_CANDIDATES, max_candidates=MAX_CANDIDATES,
        max_side=MAX_SIDE,
    )


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("image")
    if not (MIN_IMAGES <= len(files) <= MAX_IMAGES):
        raise ValidationError(
            f"原照数量须在 {MIN_IMAGES}~{MAX_IMAGES} 之间，当前 {len(files)}")

    arrays, pngs, sizes = [], [], []
    for i, fs in enumerate(files):
        arr, png, size = _load_grayscale_png(fs, f"第 {i + 1} 张原照")
        arrays.append(arr)
        pngs.append(png)
        sizes.append(size)
    if len(set(sizes)) != 1:
        raise ValidationError(f"所有裁片尺寸必须相同，当前分别为 {sizes}")
    width, height = sizes[0]

    lights = []
    for i in range(len(files)):
        lights.append(_parse_vec3(
            [request.form.get(f"light_{i}_{c}") for c in range(3)],
            f"第 {i + 1} 张原照的照明向量"))

    candidate_count = _int_field(request.form.get("candidate_count"),
                                 "备选灯数量")
    if not (MIN_CANDIDATES <= candidate_count <= MAX_CANDIDATES):
        raise ValidationError(
            f"备选灯数量须在 {MIN_CANDIDATES}~{MAX_CANDIDATES} 之间")
    candidates = []
    for j in range(candidate_count):
        candidates.append(_parse_vec3(
            [request.form.get(f"cand_{j}_{c}") for c in range(3)],
            f"第 {j + 1} 盏备选灯向量"))

    case_id = uuid.uuid4().hex
    CASES[case_id] = {
        "images": np.stack(arrays),
        "pngs": pngs,
        "size": (width, height),
        "lights": np.asarray(lights, dtype=np.float64),
        "candidates": np.asarray(candidates, dtype=np.float64),
        "created": time.time(),
        "last_result": None,
    }
    session["case_id"] = case_id
    return jsonify({"case_id": case_id, "width": width, "height": height,
                    "image_count": len(files),
                    "candidate_count": candidate_count})


@app.route("/demo")
def demo():
    """载入随仓库附带的演示裁片，进入同一套点选/求解界面。"""
    meta_path = os.path.join(app.root_path, "demo", "meta.json")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    arrays, pngs = [], []
    for name in meta["images"]:
        path = os.path.join(app.root_path, "demo", name)
        with open(path, "rb") as fh:
            data = fh.read()
        img = Image.open(io.BytesIO(data))
        img.load()
        arrays.append(np.asarray(img.convert("L"), dtype=np.uint8))
        pngs.append(data)

    case_id = uuid.uuid4().hex
    CASES[case_id] = {
        "images": np.stack(arrays),
        "pngs": pngs,
        "size": tuple(meta["size"]),
        "lights": np.asarray(meta["lights"], dtype=np.float64),
        "candidates": np.asarray(meta["candidates"], dtype=np.float64),
        "created": time.time(),
        "last_result": None,
        "demo_meta": meta,
    }
    session["case_id"] = case_id
    return render_template("case.html", case_id=case_id,
                           demo=meta, max_pairs=MAX_PAIRS)


@app.route("/case/<case_id>/image/<int:index>")
def case_image(case_id, index):
    case = CASES.get(case_id)
    if case is None or not (0 <= index < len(case["pngs"])):
        return "not found", 404
    return send_file(io.BytesIO(case["pngs"][index]), mimetype="image/png")


@app.route("/case/<case_id>")
def case_view(case_id):
    case = CASES.get(case_id)
    if case is None:
        return render_template("case.html", case_id=None), 404
    return render_template("case.html", case_id=case_id, demo=None,
                           max_pairs=MAX_PAIRS)


@app.route("/solve", methods=["POST"])
def solve():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON")

    case = CASES.get(payload.get("case_id"))
    if case is None:
        raise ValidationError("案例不存在或已过期，请重新上传")

    width, height = case["size"]
    k = case["images"].shape[0]
    pairs = _validate_pairs(payload.get("pairs"), width, height)

    reliable = payload.get("reliable")
    if (not isinstance(reliable, list) or len(reliable) != k
            or not all(isinstance(v, bool) for v in reliable)):
        raise ValidationError("可靠标记格式非法")
    if not any(reliable):
        raise ValidationError("至少保留一张可靠原照")

    scene = Scene(
        images=case["images"],
        lights=case["lights"],
        candidates=case["candidates"],
        pairs=pairs,
        reliable=reliable,
    )
    try:
        result = evaluate(scene)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": f"求解器错误: {exc}"}), 500

    # 以本次结果覆盖旧结果——推荐随最新标记/点对更新或撤下。
    case["last_result"] = {"pairs": pairs, "reliable": reliable,
                           "result": result}

    return jsonify({"ok": True, "result": _serialize_result(result, case)})


def _int_field(raw, label):
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ValidationError(f"{label} 非法: {raw!r}")


def _serialize_result(result, case):
    width, _ = case["size"]
    exclusions = [
        ("不排除" if e is None else f"排除第 {e + 1} 张原照")
        for e in result["feasible_exclusions"]
    ]
    verdicts = []
    for v in result["verdicts"]:
        verdicts.append({
            "index": v["index"],
            "vector": [round(float(x), 4) for x in case["candidates"][v["index"]]],
            "qualified": v["qualified"],
            "guarantee": (None if v["guarantee"] is None
                          else round(v["guarantee"], 4)),
            "global_min": (None if v["global_min"] is None
                           else round(v["global_min"], 4)),
            "global_max": (None if v["global_max"] is None
                           else round(v["global_max"], 4)),
            "fail_reason": v["fail_reason"],
        })
    return {
        "infeasible": result["infeasible"],
        "feasible_exclusions": exclusions,
        "verdicts": verdicts,
        "recommendation": result["recommendation"],
        "guarantee": (None if result["guarantee"] is None
                      else round(result["guarantee"], 4)),
        "gray_band": [GRAY_LOW, GRAY_HIGH],
        "gray_tol": GRAY_TOL,
    }


@app.errorhandler(ValidationError)
def handle_validation(err):
    return jsonify({"ok": False, "error": str(err)}), 400


@app.errorhandler(413)
def handle_too_large(_):
    return jsonify({"ok": False, "error": "上传内容超过 8MB 限制"}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
