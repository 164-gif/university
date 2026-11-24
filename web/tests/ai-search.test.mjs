import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {rankPassages,sourceHref} from "../ai-search-core.mjs";
import {UniversityBPETokenizer} from "../ai-search-tokenizer.mjs";
test("course, subject and duplicates",()=>{
 const rows=[{id:"a",cohort:"x",subject:"math",text:"a"},{id:"b",cohort:"y",subject:"math",text:"b"},{id:"c",cohort:"x",subject:"math",text:"a"},{id:"d",cohort:"x",subject:"physics",text:"d"}];
 assert.deepEqual(rankPassages([1,0],rows,new Float32Array([.9,0,1,0,.8,0,1,0]),2,{cohort:"x",subject:"math"}).map(x=>x.row.id),["a"]);
 assert.throws(()=>rankPassages([1],rows,new Float32Array(8),2));
});
test("source link",()=>{assert.equal(sourceHref("subjects/физика/a.md"),"./#/doc/"+encodeURIComponent("subjects/физика/a.md"));assert.throws(()=>sourceHref("subjects/../README.md"));});
test("Python/JS BPE parity",()=>{
 const tok=new UniversityBPETokenizer(JSON.parse(readFileSync(new URL("../../ai/weights/v0.2/tokenizer.json",import.meta.url),"utf8")));
 const fixtures=JSON.parse(readFileSync(new URL("../.ai-tokenizer-fixtures.json",import.meta.url),"utf8"));
 for(const sample of fixtures.samples)assert.deepEqual(tok.encode(sample.text),sample.ids,sample.text);
});
