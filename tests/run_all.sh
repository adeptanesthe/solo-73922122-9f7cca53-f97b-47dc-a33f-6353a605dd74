#!/usr/bin/env bash
# 全套自测：求解器 LP 逻辑 + Flask 端到端 + 前端 DOM 交互。
# 前置：Python 依赖已安装（pip install -r requirements.txt），Node 可用且
# tests 目录下已 npm install jsdom（或设置 NODE_PATH 指向其安装位置）。
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

$PY tests/test_solver.py
$PY tests/test_app.py

# DOM 夹具是真实渲染的演示页，每次按当前模板重新生成
$PY - <<'PYEOF'
from app import app
with app.test_client() as c:
    html = c.get('/demo').get_data(as_text=True)
open('tests/dom_fixture.html', 'w', encoding='utf-8').write(html)
print('已生成 tests/dom_fixture.html')
PYEOF

node tests/dom_test.js
