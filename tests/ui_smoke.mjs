/* Headless smoke test of the UI against a live server using jsdom.
 *
 * Usage: node tests/ui_smoke.mjs http://127.0.0.1:8123
 * Requires jsdom available from the working directory (see README).
 */
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

// jsdom is an optional dev dependency; point JSDOM_DIR at a node_modules
// directory that contains it when it is not installed next to this file.
const require = createRequire(
  process.env.JSDOM_DIR ? path.join(process.env.JSDOM_DIR, "noop.js") : import.meta.url,
);
const { JSDOM, VirtualConsole } = require("jsdom");

const BASE = process.argv[2] || "http://127.0.0.1:8123";
const root = path.resolve(new URL("..", import.meta.url).pathname);

const html = fs.readFileSync(path.join(root, "app/static/index.html"), "utf8");
const script = fs.readFileSync(path.join(root, "app/static/app.js"), "utf8");

const virtualConsole = new VirtualConsole();
const errors = [];
virtualConsole.on("jsdomError", (e) => errors.push(String(e)));
virtualConsole.on("error", (e) => errors.push(String(e)));

const dom = new JSDOM(html, {
  url: BASE + "/",
  runScripts: "outside-only",
  virtualConsole,
  pretendToBeVisual: true,
});
const { window } = dom;
window.fetch = (input, init) => fetch(new URL(input, BASE), init);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(fn, label, timeout = 20000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (fn()) return;
    await sleep(50);
  }
  throw new Error("timeout waiting for " + label);
}
const $ = (id) => window.document.getElementById(id);

window.eval(script);
await sleep(200);

const results = [];
function check(name, condition, detail = "") {
  results.push({ name, ok: !!condition, detail });
  console.log(`${condition ? "PASS" : "FAIL"}  ${name}${detail ? " — " + detail : ""}`);
}

// 1. samples are listed
await until(() => $("sample-buttons").children.length > 0, "sample buttons");
check("样例按钮已渲染", $("sample-buttons").children.length >= 3,
  $("sample-buttons").textContent.trim());

// 2. load the repeat sample
const repeatBtn = [...$("sample-buttons").children].find((b) => b.textContent.includes("sample_repeat"));
repeatBtn.click();
await until(() => !$("workspace").hidden && $("progress-text").textContent !== "0 / 0 帧", "sample loaded");
check("载入样例后显示工作区", !$("workspace").hidden, $("progress-text").textContent);
check("初始无候选字幕", $("confirmed-text").textContent === "", `“${$("confirmed-text").textContent}”`);

// 3. advance one chunk
window.document.getElementById("chunk-size").value = "5";
$("chunk-size").dispatchEvent(new window.Event("input"));
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("5 /"), "advance chunk");
check("分块推进更新进度", $("progress-text").textContent.startsWith("5 /"), $("progress-text").textContent);
check("候选表非空", $("beam-body").children.length > 0, `${$("beam-body").children.length} 束`);
const firstText = $("beam-body").children[0].querySelector(".beam-text").textContent;
check("最佳候选含字符", firstText.length > 0, firstText);

// 4. confirm the best candidate
$("confirm-best").click();
await until(() => $("confirmed-text").textContent.length > 0, "confirm");
const confirmed = $("confirmed-text").textContent;
check("确认后前缀显示", confirmed.length > 0, confirmed);
check("blank 分数显示", $("ps-blank").textContent !== "—", $("ps-blank").textContent);
check("非 blank 分数显示", $("ps-nonblank").textContent !== "—", $("ps-nonblank").textContent);
check("撤销按钮可用", !$("undo-btn").disabled);

// 5. advance after confirmation, then undo
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("10 /"), "advance after confirm");
$("undo-btn").click();
await until(() => $("confirmed-text").textContent === "", "undo");
check("撤销后前缀清空", $("confirmed-text").textContent === "");
check("撤销后进度保留", $("progress-text").textContent.startsWith("10 /"), $("progress-text").textContent);
check("撤销后候选非空", $("beam-body").children.length > 0);

// 6. failure keeps state: confirm text not in beam
$("prefix-input").value = "zzzz";
$("apply-prefix").click();
await until(() => !$("op-error").hidden, "error shown");
const beforeProgress = $("progress-text").textContent;
check("非法确认显示错误", !$("op-error").hidden, $("op-error").textContent);
check("错误后进度不变", $("progress-text").textContent === beforeProgress, $("progress-text").textContent);

// 7. historical frame inspection
$("peek-frame").value = "2";
$("peek-frame").dispatchEvent(new window.Event("input"));
await sleep(400);
check("历史帧提示只读", $("peek-note").textContent.includes("历史回放"), $("peek-note").textContent);
check("历史帧概率已渲染", $("frame-probs").children.length > 0);

// 8. continuous run to the end
$("peek-frame").value = $("peek-frame").max;
$("peek-frame").dispatchEvent(new window.Event("input"));
$("run-btn").click();
await until(() => $("progress-text").textContent.startsWith("15 / 15"), "run to end", 30000);
check("连续推进到结尾", $("progress-text").textContent.startsWith("15 / 15"), $("progress-text").textContent);
await until(() => $("stop-btn").hidden, "run loop finished");
check("运行结束停止按钮隐藏", $("stop-btn").hidden);

// 8.5 reference-subtitle alignment on the fully advanced sample
$("align-input").value = "aab";
$("align-btn").click();
await until(() => !$("align-result").hidden && $("align-body").children.length === 3, "align result");
check("对齐结果逐字成行", $("align-body").children.length === 3, `${$("align-body").children.length} 行`);
check("结果保留原文本与帧号",
  $("align-summary").textContent.includes("「aab」") && $("align-summary").textContent.includes("第 15 帧边界"),
  $("align-summary").textContent);
check("新鲜结果无过期标记", $("align-stale").hidden);
const starts = [...$("align-body").children].map((tr) => Number(tr.children[2].textContent));
const ends = [...$("align-body").children].map((tr) => Number(tr.children[3].textContent));
check("起止帧为合法帧号",
  starts.every((v, i) => v >= 0 && v <= ends[i] && ends[i] < 15) && starts.every((v, i) => i === 0 || v > starts[i - 1]),
  `起帧 ${starts} 止帧 ${ends}`);
check("逐字路径 log P 有数值", [...$("align-body").children].every((tr) => tr.children[4].textContent !== "—"));

// editing the reference text marks the kept result stale, restoring clears it
$("align-input").value = "ab";
$("align-input").dispatchEvent(new window.Event("input"));
check("改动参考文本后旧结果标为过期", !$("align-stale").hidden);
$("align-input").value = "aab";
$("align-input").dispatchEvent(new window.Event("input"));
check("恢复文本后过期标记消失", $("align-stale").hidden);

// 8.6 advancing frames marks the kept alignment stale
const blankBtn = [...$("sample-buttons").children].find((b) => b.textContent.includes("blank_heavy"));
blankBtn.click();
await until(() => $("progress-text").textContent.startsWith("0 / 21"), "blank heavy loaded");
check("新会话清空旧对齐结果", $("align-result").hidden);
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("5 / 21"), "advance 5");
$("align-input").value = "啊";
$("align-btn").click();
await until(() => !$("align-result").hidden && $("align-body").children.length === 1, "align 啊");
check("推进前对齐结果新鲜", $("align-stale").hidden);
const summaryBefore = $("align-summary").textContent;
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("10 / 21"), "advance 10");
check("推进帧后旧结果标为过期", !$("align-stale").hidden);
check("过期结果仍保留原文本与帧号",
  $("align-summary").textContent === summaryBefore && $("align-summary").textContent.includes("第 5 帧边界"),
  $("align-summary").textContent);
$("align-btn").click();
await until(() => $("align-stale").hidden && $("align-summary").textContent.includes("第 10 帧边界"), "re-aligned");
check("重新对齐后结果刷新", $("align-summary").textContent.includes("第 10 帧边界"), $("align-summary").textContent);

// 9. zero-probability sample shows -∞ without NaN
const zeroBtn = [...$("sample-buttons").children].find((b) => b.textContent.includes("sparse_zero"));
zeroBtn.click();
await until(() => $("progress-text").textContent.startsWith("0 / 7"), "zero sample loaded");
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("7 / 7") || $("progress-text").textContent.startsWith("5 / 7"), "zero sample advance");
await sleep(300);
const bodyText = $("beam-body").textContent;
check("零概率样例无 NaN", !bodyText.includes("NaN"), bodyText.slice(0, 120));
check("零概率样例显示 -∞", bodyText.includes("-∞") || $("frame-probs").textContent.includes("0.000"));

// 9.5 unalignable reference shows a hint and keeps the (empty) result area
$("align-input").value = "z";
$("align-btn").click();
await until(() => !$("align-notice").hidden, "align hint");
check("无法完整对齐时给出提示", $("align-notice").textContent.includes("无法完整对齐"), $("align-notice").textContent.slice(0, 70));
check("提示时不显示结果表", $("align-result").hidden);
$("align-input").value = "xy";
$("align-btn").click();
await until(() => !$("align-result").hidden && $("align-body").children.length === 2, "align xy");
check("可对齐文本显示逐字表格", $("align-body").children.length === 2);
check("对齐成功后提示消失", $("align-notice").hidden);

// 10. no uncaught UI errors
check("无未捕获脚本错误", errors.length === 0, errors.join(" | ").slice(0, 200));

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
