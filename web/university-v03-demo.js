import * as ort from "./runtime/ort.wasm.min.mjs";
import { UniversityBPETokenizer } from "./ai-search-tokenizer.js";

const status = document.querySelector("#status");
const button = document.querySelector("#submit");
const results = document.querySelector("#results");
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./runtime/", import.meta.url).href;

async function load(name, type) {
  const response = await fetch(new URL(name, import.meta.url));
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  return response[type]();
}

let context;
try {
  const [configuration, manifest, rows, index] = await Promise.all([
    load("tokenizer.json", "json"), load("manifest.json", "json"),
    load("passages.json", "json"), load("embeddings.f32", "arrayBuffer"),
  ]);
  const embeddings = new Float32Array(index);
  if (rows.length !== manifest.rows || embeddings.length !== manifest.rows * manifest.dimensions) {
    throw new Error("Размер индекса не совпадает с манифестом.");
  }
  const session = await ort.InferenceSession.create(new URL("encoder.onnx", import.meta.url).href, {
    executionProviders: ["wasm"],
  });
  context = { tokenizer: new UniversityBPETokenizer(configuration), session, rows, embeddings, manifest };
  button.disabled = false;
  status.textContent = `Готово. Фрагментов в индексе: ${rows.length}.`;
} catch (error) {
  status.textContent = `Не удалось загрузить поиск. ${error.message}`;
}

function input(text, tokenizer) {
  const encoded = tokenizer.encode(text);
  return encoded.length <= 512 ? encoded : [...encoded.slice(0, 511), 2];
}

async function vector(text) {
  const ids = input(text, context.tokenizer);
  const output = await context.session.run({
    input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]),
    attention_mask: new ort.Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
  });
  return output.embedding.data;
}

function rank(embedding) {
  const { rows, embeddings, manifest } = context;
  if (embedding.length !== manifest.dimensions) throw new Error("Размер вектора модели отличается от индекса.");
  return rows.map((row, index) => {
    let score = 0;
    for (let i = 0; i < embedding.length; i++) score += embedding[i] * embeddings[index * embedding.length + i];
    return { row, score };
  }).sort((a, b) => b.score - a.score).slice(0, 5);
}

document.querySelector("#search").addEventListener("submit", async event => {
  event.preventDefault();
  const query = document.querySelector("#query").value.trim();
  if (!query || !context) return;
  button.disabled = true;
  results.replaceChildren();
  const start = performance.now();
  try {
    for (const { row } of rank(await vector(query))) {
      const item = document.createElement("li");
      const paragraph = document.createElement("p");
      paragraph.textContent = row.text;
      const link = document.createElement("a");
      const path = row.source.split("/").map(encodeURIComponent).join("/");
      link.href = `https://github.com/164-gif/university/blob/${context.manifest.source_commit}/${path}#L${row.line_start}-L${row.line_end}`;
      link.textContent = `${row.subject} · источник, строки ${row.line_start}–${row.line_end}`;
      item.append(paragraph, link);
      results.append(item);
    }
    status.textContent = `Поиск выполнен за ${Math.round(performance.now() - start)} мс.`;
  } catch (error) {
    status.textContent = `Ошибка поиска. ${error.message}`;
  } finally {
    button.disabled = false;
  }
});
