/** Run the exported v0.3 bundle in Chromium through the local DevTools protocol. */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import http from "node:http";
import { tmpdir } from "node:os";
import { extname, join, resolve, sep } from "node:path";
import { setTimeout } from "node:timers/promises";

const root = resolve(process.argv[2]);
const profile = await mkdtemp(join(tmpdir(), "university-v03-chromium-"));
const types = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".wasm": "application/wasm", ".json": "application/json",
};

const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, "http://localhost");
    const path = resolve(root, "." + decodeURIComponent(url.pathname === "/" ? "/index.html" : url.pathname));
    if (!path.startsWith(root + sep) || !(await stat(path)).isFile()) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, { "Content-Type": types[extname(path)] || "application/octet-stream" });
    createReadStream(path).pipe(response);
  } catch {
    response.writeHead(404).end();
  }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const chromium = spawn(process.env.CHROMIUM || "chromium", [
  "--headless=new", "--disable-gpu", "--disable-background-networking", "--no-first-run",
  "--remote-debugging-port=0", "--user-data-dir=" + profile, origin,
], { stdio: "ignore" });

async function until(check, timeout = 30000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const value = await check();
    if (value) return value;
    await setTimeout(100);
  }
  throw new Error("Timed out waiting for Chromium.");
}

let socket;
try {
  const port = await until(async () => {
    try { return Number((await readFile(join(profile, "DevToolsActivePort"), "utf8")).split("\n")[0]); }
    catch { return undefined; }
  });
  const tab = await until(async () => {
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    return targets.find(item => item.type === "page" && item.url.startsWith(origin));
  });
  socket = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onerror = reject;
  });
  let lastId = 0;
  const pending = new Map();
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    }
  };
  async function evaluate(expression) {
    const id = ++lastId;
    const pendingResult = new Promise(resolve => pending.set(id, resolve));
    socket.send(JSON.stringify({ id, method: "Runtime.evaluate",
      params: { expression, awaitPromise: true, returnByValue: true } }));
    const { result, error } = await pendingResult;
    if (error || result.exceptionDetails) throw new Error(JSON.stringify(error || result.exceptionDetails));
    return result.result.value;
  }
  await until(() => evaluate(`document.querySelector("#submit") && !document.querySelector("#submit").disabled`));
  const results = await evaluate(`(async () => {
    const [config, probes] = await Promise.all([
      fetch("./tokenizer.json").then(r => r.json()), fetch("./tokenizer-parity.json").then(r => r.json()),
    ]);
    const { UniversityBPETokenizer } = await import("./ai-search-tokenizer.js");
    const tokenizer = new UniversityBPETokenizer(config);
    const parity = probes.filter(p => p.text).every(p => JSON.stringify(p.ids) === JSON.stringify(tokenizer.encode(p.text)));
    const { env, InferenceSession, Tensor } = await import("./runtime/ort.wasm.min.mjs");
    env.wasm.numThreads = 1;
    env.wasm.wasmPaths = new URL("./runtime/", location.href).href;
    const session = await InferenceSession.create("./encoder.onnx", { executionProviders: ["wasm"] });
    const shapes = [];
    let maxError = 0;
    for (const length of [7, 13]) {
      const ids = Array.from({length: 2*length}, (_, i) => 1n+BigInt(i % 31));
      const mask = new BigInt64Array(2*length).fill(1n);
      const { embedding } = await session.run({input_ids: new Tensor("int64", BigInt64Array.from(ids), [2,length]),
        attention_mask: new Tensor("int64",mask,[2,length])});
      shapes.push(embedding.dims);
      if (!embedding.data.every(Number.isFinite)) throw new Error("Non-finite browser output");
      const reference = probes.find(p => p.length === length)?.embedding || [];
      for (let i=0; i<reference.length; i++) maxError = Math.max(maxError, Math.abs(reference[i]-embedding.data[i]));
    }
    const query = document.querySelector("#query");
    const form = document.querySelector("#search");
    const button = document.querySelector("#submit");
    const elapsed = [];
    for (let i=0; i<20; i++) {
      query.value = probes[i % 5].text;
      const start = performance.now();
      form.requestSubmit();
      while (button.disabled) await new Promise(r => setTimeout(r,1));
      if (document.querySelectorAll("#results li").length !== 5) throw new Error(document.querySelector("#status").textContent);
      elapsed.push(performance.now()-start);
    }
    elapsed.sort((a,b)=>a-b);
    return { parity, shapes, maxError, latency_p50_ms: elapsed[9], latency_p95_ms: elapsed[18],
      results: document.querySelectorAll("#results li").length,
      source: document.querySelector("#results a").href,
      resources: [...new Set(performance.getEntriesByType("resource").map(r => new URL(r.name).origin))],
      chromium: navigator.userAgent };
  })()`);
  assert.equal(results.parity, true);
  assert.ok(results.maxError < 1e-3, `CPU/WASM embedding delta: ${results.maxError}`);
  assert.deepEqual(results.resources, [origin]);
  assert.equal(results.shapes.length, 2);
  assert.equal(results.shapes[0][0], 2);
  assert.match(results.source, /github\.com\/164-gif\/university/);
  const report = { ...results, mode: "Chromium ONNX Runtime Web WASM, single thread" };
  await writeFile(resolve(root, "../browser-report.json"), JSON.stringify(report, null, 2) + "\n");
  console.log(report);
} finally {
  socket?.close();
  const exited = new Promise(resolve => chromium.once("exit", resolve));
  chromium.kill();
  server.close();
  await Promise.race([exited, setTimeout(2000)]);
  await rm(profile, { recursive: true, force: true, maxRetries: 3 });
}
