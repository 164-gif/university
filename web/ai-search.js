import * as ort from "./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs";
import { UniversityBPETokenizer } from "./ai-search-tokenizer.js";

ort.env.wasm.wasmPaths = new URL("./node_modules/onnxruntime-web/dist/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const root = "../ai/generated/full";
const status = document.querySelector("#status");
const results = document.querySelector("#results");

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error("Не удалось загрузить " + path + ": " + response.status);
  return response.json();
}

const [tokenizerJson, passages, manifest] = await Promise.all([
  loadJson(root + "/tokenizer.json"), loadJson(root + "/browser/passages.json"),
  loadJson(root + "/browser/manifest.json")
]);
const tokenizer = new UniversityBPETokenizer(tokenizerJson);
const embeddingResponse = await fetch(root + "/browser/embeddings.f32");
if (!embeddingResponse.ok) throw new Error("Не удалось загрузить индекс эмбеддингов");
const embeddingBuffer = await embeddingResponse.arrayBuffer();
const embeddings = new Float32Array(embeddingBuffer);
if (passages.length !== manifest.rows || embeddings.length !== manifest.rows * manifest.dimensions) {
  throw new Error("Файлы индекса не соответствуют manifest.json");
}
status.textContent = "Загрузка модели и " + passages.length + " фрагментов…";
const session = await ort.InferenceSession.create(root + "/trained/encoder-v0.2.onnx", {
  executionProviders: ["wasm"], graphOptimizationLevel: "all"
});
status.textContent = "Готово. Индекс: " + passages.length + " фрагментов.";

async function search(text) {
  const encoded = tokenizer.inputs(text);
  const sequence = encoded.input_ids.length;
  const ids = new BigInt64Array(encoded.input_ids.map(BigInt));
  const mask = new BigInt64Array(encoded.attention_mask.map(BigInt));
  const output = await session.run({
    input_ids: new ort.Tensor("int64", ids, [1, sequence]),
    attention_mask: new ort.Tensor("int64", mask, [1, sequence])
  });
  const query = output.embedding.data;
  const hits = [];
  for (let i = 0; i < passages.length; i++) {
    let score = 0;
    const offset = i * manifest.dimensions;
    for (let d = 0; d < query.length; d++) score += query[d] * embeddings[offset + d];
    hits.push({ row: passages[i], score });
  }
  return hits.sort((a, b) => b.score - a.score).slice(0, 5);
}

document.querySelector("#search").addEventListener("submit", async event => {
  event.preventDefault();
  const query = document.querySelector("#query").value.trim();
  if (!query) return;
  results.replaceChildren();
  status.textContent = "Ищу…";
  try {
    const hits = await search(query);
    for (const hit of hits) {
      const item = document.createElement("li");
      const body = document.createElement("div");
      body.textContent = hit.row.text;
      const meta = document.createElement("small");
      meta.textContent = hit.row.subject + " · cosine " + hit.score.toFixed(3);
      item.append(body, meta);
      results.append(item);
    }
    status.textContent = "Поиск выполнен локально. У этой экспериментальной модели качество ниже BM25F.";
  } catch (error) {
    status.textContent = "Ошибка поиска: " + error.message;
  }
});
