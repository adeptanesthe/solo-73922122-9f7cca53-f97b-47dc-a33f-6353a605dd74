// 前端 DOM 交互自测：node tests/dom_test.js
// 用 jsdom 加载真实渲染的演示页，桩掉 fetch / canvas / Image.decode，
// 验证点选、求解渲染、改标重算、无共同解释与无灯可用的 UI 文案。
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('/tmp/pw/node_modules/jsdom');

const BASE = 'http://127.0.0.1:8000';
const html = fs.readFileSync(path.join(__dirname, 'dom_fixture.html'), 'utf8');

const tick = (ms = 30) => new Promise(r => setTimeout(r, ms));

function makeDom(handler) {
  // 页面初始化期间就会发起 fetch，处理器经 beforeParse 提前挂好。
  const holder = { handler };
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: BASE + '/demo',
    beforeParse(window) {
      // jsdom 不实现 canvas 2d 上下文与图像解码。
      window.HTMLCanvasElement.prototype.getContext = () => ({
        clearRect() {}, setTransform() {}, scale() {}, beginPath() {},
        arc() {}, fill() {}, fillText() {},
      });
      // 桩接图像：decode 立即兑现，自然尺寸固定为演示裁片的 4x4
      window.HTMLImageElement.prototype.decode = function () {
        Object.defineProperty(this, 'naturalWidth',
          { value: 4, configurable: true });
        Object.defineProperty(this, 'naturalHeight',
          { value: 4, configurable: true });
        return Promise.resolve();
      };
      window.fetch = (url, opts) => holder.handler(url, opts);
    },
  });
  return { dom, holder };
}

const asserts = [];
function check(cond, msg) {
  if (!cond) throw new Error(msg);
  asserts.push('  ✓ ' + msg);
}

function jsonResponse(body, status = 200) {
  return {
    ok: status < 400, status,
    url: BASE + '/x',
    json: async () => body,
  };
}

const okResult = {
  ok: true,
  result: {
    infeasible: false,
    feasible_exclusions: ['不排除'],
    gray_band: [5, 250], gray_tol: 2,
    recommendation: 0, guarantee: 50.04,
    verdicts: [
      { index: 0, vector: [0, 0, 0.9], qualified: true, guarantee: 50.04,
        global_min: 160, global_max: 218, fail_reason: null },
      { index: 1, vector: [0.7, 0, 0.7], qualified: true, guarantee: -54.5,
        global_min: 119, global_max: 216, fail_reason: null },
      { index: 2, vector: [1.5, 0, 0.9], qualified: false, guarantee: null,
        global_min: null, global_max: null,
        fail_reason: '像素(1,1) 预测灰度可达 148.08 ~ 355.92，超出 [5, 250]' },
      { index: 3, vector: [0.1, 0.1, 0.8], qualified: true, guarantee: 32.3,
        global_min: 154, global_max: 194, fail_reason: null },
    ],
  },
};

async function run() {
  // ---- 场景一：正常演示，自动求解并渲染推荐 ----
  const solveCalls = [];
  const { dom } = makeDom((url, opts) => {
    if (url.includes('/image/')) {
      const idx = Number(url.split('/image/')[1]);
      return Promise.resolve(jsonResponse({}, idx < 4 ? 200 : 404));
    }
    solveCalls.push(opts ? JSON.parse(opts.body) : null);
    return Promise.resolve(jsonResponse(okResult));
  });
  const { window } = dom;
  await tick();

  check(solveCalls.length === 1, '演示页加载后自动求解一次');
  check(JSON.stringify(solveCalls[0].pairs) ===
        JSON.stringify([[[1, 1], [0, 0]], [[2, 1], [3, 1]]]),
        '自动求解携带演示建议点对');
  const resultEl = window.document.getElementById('result');
  check(resultEl.textContent.includes('推荐：第 1 盏'), '渲染推荐横幅');
  check(resultEl.textContent.includes('50.04'), '显示保证值');
  check(window.document.querySelectorAll('.rec-row').length === 1,
        '推荐行高亮且唯一');
  check(window.document.querySelectorAll('#thumbs figure').length === 4,
        '渲染 4 张原照缩略图与可靠复选框');

  // ---- 改标后自动重算（change 事件）----
  const checkbox = window.document.querySelector('input[data-img="0"]');
  checkbox.checked = false;
  checkbox.dispatchEvent(new window.Event('change', { bubbles: true }));
  await tick();
  check(solveCalls.length === 2, '更改可靠标记后自动重新求解');
  check(solveCalls[1].reliable[0] === false, '重算请求带上新的可靠标记');

  // ---- 场景二：无共同解释，撤下推荐 ----
  const infeasible = {
    ok: true,
    result: {
      infeasible: true, feasible_exclusions: [],
      recommendation: null, guarantee: null,
      verdicts: [0, 1, 2, 3].map(i => ({
        index: i, vector: [0, 0, 0], qualified: false, guarantee: null,
        global_min: null, global_max: null,
        fail_reason: '原照无共同解释，无法评估',
      })),
    },
  };
  const d2 = makeDom((url) =>
    url.includes('/image/')
      ? Promise.resolve(jsonResponse({},
          Number(url.split('/image/')[1]) < 4 ? 200 : 404))
      : Promise.resolve(jsonResponse(infeasible)));
  await tick();
  const txt2 = d2.dom.window.document.getElementById('result').textContent;
  check(txt2.includes('原照无共同解释'), '无共同解释时显示提示');
  check(!txt2.includes('推荐：第'), '无共同解释时撤下旧推荐');

  // ---- 场景三：有解释但无灯可用 ----
  const noLight = {
    ok: true,
    result: {
      infeasible: false, feasible_exclusions: ['不排除'],
      recommendation: null, guarantee: null,
      verdicts: [0, 1].map(i => ({
        index: i, vector: [5, 0, 3], qualified: false, guarantee: null,
        global_min: null, global_max: null,
        fail_reason: '像素(0,0) 预测灰度可达 255.9 ~ 330.6，超出 [5, 250]',
      })),
    },
  };
  const d3 = makeDom((url) =>
    url.includes('/image/')
      ? Promise.resolve(jsonResponse({},
          Number(url.split('/image/')[1]) < 4 ? 200 : 404))
      : Promise.resolve(jsonResponse(noLight)));
  await tick();
  const txt3 = d3.dom.window.document.getElementById('result').textContent;
  check(txt3.includes('无灯可用'), '无灯可用时显示提示');
  check(!txt3.includes('推荐：第'), '无灯可用时撤下旧推荐');

  // ---- 点对交互：添加一对、重复像素被拒 ----
  const d4 = makeDom((url) =>
    url.includes('/image/')
      ? Promise.resolve(jsonResponse({},
          Number(url.split('/image/')[1]) < 4 ? 200 : 404))
      : Promise.resolve(jsonResponse(infeasible)));
  await tick();
  const w4 = d4.dom.window;
  w4.document.getElementById('start-pair').click();
  check(w4.document.getElementById('pick-hint').textContent.includes('刻线'),
        '进入刻线点选状态');
  const canvas = w4.document.getElementById('pick-canvas');
  function clickAt(x, y) {
    const rect = { left: 0, top: 0 };
    canvas.getBoundingClientRect = () => ({ ...rect });
    const ev = new w4.MouseEvent('click', {
      bubbles: true,
      clientX: rect.left + x * 32 + 4,
      clientY: rect.top + y * 32 + 4,
    });
    canvas.dispatchEvent(ev);
  }
  clickAt(2, 2);
  check(w4.document.getElementById('pick-hint').textContent.includes('底面'),
        '刻线选中后进入底面点选状态');
  clickAt(2, 2);
  check(w4.document.getElementById('pick-hint').textContent.includes('不同'),
        '同像素两点被拒绝并提示');
  clickAt(3, 3);
  check(w4.document.querySelectorAll('#pair-list li').length === 3,
        '演示两对 + 新点选一对 = 3 对（上限 4）');
  check(w4.document.getElementById('start-pair').hidden === false,
        '成对点选完成后恢复添加按钮');

  console.log(asserts.join('\n'));
  console.log('\nDOM 交互自测全部通过。');
}

run().catch(err => { console.error(err); process.exit(1); });
