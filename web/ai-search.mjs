import {UniversityBPETokenizer} from "./ai-search-tokenizer.mjs";
import {rankPassages,sourceHref} from "./ai-search-core.mjs";
try{document.documentElement.dataset.theme=localStorage.getItem("university-theme")||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");}catch{}
const root=new URL("./ai-search-assets/",import.meta.url);
const $=id=>document.getElementById(id);
const form=$("ai-form"),button=$("ai-submit"),status=$("ai-status"),results=$("ai-results"),cohort=$("ai-cohort"),subject=$("ai-subject");
const names={analytical_geometry:"Аналитическая геометрия",mathematical_analysis:"Математический анализ",physics:"Физика",laboratory_physics:"Лабораторные работы по физике",obzh:"ОБЖ",introduction_to_information_security:"Введение в информационную безопасность",russian_statehood_fundamentals:"Основы российской государственности",english:"Английский язык",informatics:"Информатика"};
const cohorts={"ib-2026-1kurs":"ИБ-2026 — 1 курс","ib-2025-1kurs":"ИБ-2025 — 1 курс","ib-2026-2kurs":"ИБ-2026 — 2 курс"};
let loaded;
async function fetchOk(name,hash){const url=new URL(name,root);if(hash)url.searchParams.set("v",hash);const r=await fetch(url,{cache:hash?"default":"no-cache"});if(!r.ok)throw new Error("Не удалось загрузить "+name+" ("+r.status+")");return r;}
async function verified(name,manifest){
  const entry=manifest.files[name];
  if(!entry||! /^[a-f0-9]{64}$/.test(entry.sha256))throw new Error("Нет контрольной суммы "+name);
  const bytes=await(await fetchOk(name,entry.sha256)).arrayBuffer();
  if(bytes.byteLength!==entry.bytes)throw new Error("Неверный размер "+name);
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",bytes)),b=>b.toString(16).padStart(2,"0")).join("");
  if(hash!==entry.sha256)throw new Error("Неверная контрольная сумма "+name);
  return bytes;
}
const decodeJson=bytes=>JSON.parse(new TextDecoder().decode(bytes));
function fillOptions(select,items,selected){select.replaceChildren(...items.map(([value,label])=>new Option(label,value)));if(items.some(([value])=>value===selected))select.value=selected;}
function fillSubjects(){
  const ids=[...new Set(loaded.rows.filter(r=>r.cohort===cohort.value).map(r=>r.subject))].sort();
  fillOptions(subject,[["","Все предметы"],...ids.map(id=>[id,names[id]||id])],subject.value);
}
async function init(){
  status.textContent="Загрузка модели и индекса (первый запуск может занять время)…";
  const manifest=await(await fetchOk("manifest.json")).json();
  if(manifest.format!=="university-browser-index-v1"||manifest.model_version!=="v0.2"||manifest.dimensions!==128||manifest.dtype!=="float32-little-endian"||!Number.isInteger(manifest.rows)||manifest.rows<1)throw new Error("Несовместимый индекс");
  const [tok,passages,emb,model,ort]=await Promise.all([verified("tokenizer.json",manifest),verified("passages.json",manifest),verified("embeddings.f32",manifest),verified("encoder.onnx",manifest),import("./ai-search-assets/runtime/ort.wasm.min.mjs")]);
  const tokenizer=new UniversityBPETokenizer(decodeJson(tok)),rows=decodeJson(passages),vectors=new Float32Array(emb);
  if(rows.length!==manifest.rows||vectors.length!==rows.length*128)throw new Error("Размеры индекса не совпадают");
  ort.env.wasm.wasmPaths=new URL("runtime/",root).href;ort.env.wasm.numThreads=1;
  status.textContent="Инициализация ONNX Runtime…";
  const session=await ort.InferenceSession.create(model,{executionProviders:["wasm"],graphOptimizationLevel:"all"});
  loaded={manifest,tokenizer,rows,vectors,ort,session};
  const ids=[...new Set(rows.map(r=>r.cohort))].sort();
  fillOptions(cohort,ids.map(id=>[id,cohorts[id]||id]),cohort.value);fillSubjects();
  button.disabled=cohort.disabled=subject.disabled=false;
  status.textContent="Модель готова. В индексе "+rows.length+" фрагментов.";
}
function fail(error){status.textContent="ИИ-поиск недоступен: "+error.message+". Обычный поиск продолжает работать.";}
cohort.addEventListener("change",fillSubjects);
form.addEventListener("submit",async event=>{
  event.preventDefault();const question=$("ai-query").value.trim();
  if(!question||!loaded||button.disabled)return;
  button.disabled=cohort.disabled=subject.disabled=true;results.replaceChildren();
  try{
    const {manifest,tokenizer,rows,vectors,ort,session}=loaded;
    status.textContent="Ищу подходящие фрагменты…";const t=performance.now();
    const {input_ids,attention_mask}=tokenizer.inputs(question,manifest.max_length);
    const tensor=ids=>new ort.Tensor("int64",BigInt64Array.from(ids,BigInt),[1,ids.length]);
    const output=await session.run({input_ids:tensor(input_ids),attention_mask:tensor(attention_mask)});
    const hits=rankPassages(output.embedding.data,rows,vectors,128,{cohort:cohort.value,subject:subject.value});
    for(const {row,score} of hits){
      const item=document.createElement("li");item.className="ai-result";
      const heading=document.createElement("h2"),link=document.createElement("a");
      link.href=sourceHref(row.source);link.textContent=row.headings?.at(-1)||row.title;heading.append(link);
      const meta=document.createElement("div");meta.className="ai-meta";meta.textContent=(names[row.subject]||row.subject)+" · "+row.layer+" · строки "+row.line_start+"–"+row.line_end+" · сходство "+score.toFixed(3);
      const body=document.createElement("div");body.className="ai-excerpt";body.textContent=row.text;
      item.append(heading,meta,body);
      if(row.document_warning){const warning=document.createElement("div");warning.className="ai-warning";warning.textContent="В источнике есть места, требующие проверки.";item.append(warning);}
      results.append(item);
    }
    status.textContent=hits.length?"Кандидаты: "+hits.length+" · "+Math.round(performance.now()-t)+" мс · без проверенного порога отказа.":"В выбранном разделе нет доступных фрагментов.";
  }catch(error){fail(error);}finally{button.disabled=cohort.disabled=subject.disabled=false;}
});
init().catch(fail);
