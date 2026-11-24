import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {rankPassages,sourceHref,LexicalIndex,hybridRank,near,lexicalTokens} from "../ai-search-core.mjs";
import {UniversityBPETokenizer} from "../ai-search-tokenizer.mjs";
test("course, subject and duplicates",()=>{
 const rows=[{id:"a",cohort:"x",subject:"math",text:"a"},{id:"b",cohort:"y",subject:"math",text:"b"},{id:"c",cohort:"x",subject:"math",text:"a"},{id:"d",cohort:"x",subject:"physics",text:"d"}];
 assert.deepEqual(rankPassages([1,0],rows,new Float32Array([.9,0,1,0,.8,0,1,0]),2,{cohort:"x",subject:"math"}).map(x=>x.row.id),["a"]);
 assert.throws(()=>rankPassages([1],rows,new Float32Array(8),2));
});
test("source link",()=>{assert.equal(sourceHref("subjects/физика/a.md"),"./#/doc/"+encodeURIComponent("subjects/физика/a.md"));assert.throws(()=>sourceHref("subjects/../README.md"));});
test("Python/JS BPE parity",()=>{
 const tok=new UniversityBPETokenizer(JSON.parse(readFileSync(new URL("../../_site/ai-search-assets/tokenizer.json",import.meta.url),"utf8")));
 const fixtures=JSON.parse(readFileSync(new URL("../.ai-tokenizer-fixtures.json",import.meta.url),"utf8"));
 for(const sample of fixtures.samples)assert.deepEqual(tok.encode(sample.text),sample.ids,sample.text);
});

test("lexical normalization and typo types",()=>{
 assert.deepEqual(lexicalTokens("Что такое скалярное произведение?"),lexicalTokens("СКАЛЯРНЫМ произведением"));
 assert.ok(near("сколярн","скалярн"));
 assert.ok(near("скалярн","скалрян"));
 assert.ok(near("произведен","произведн"));
 assert.ok(!near("скалярн","векторн"));
});
const rows=[
 {id:"scalar",cohort:"x",subject:"math",title:"Скалярное произведение",headings:["Скалярное произведение"],text:"Скалярное произведение равно произведению модулей на косинус угла."},
 {id:"vector",cohort:"x",subject:"math",title:"Векторное произведение",headings:["Векторное произведение"],text:"Векторное произведение двух векторов."},
 {id:"other",cohort:"y",subject:"math",title:"Скалярное произведение",headings:["Скалярное произведение"],text:"Другой курс."}
];
test("scoped typo search and lexical priority",()=>{
 const index=new LexicalIndex(rows);
 for(const q of ["скалярное произведение","сколярное произведение","скалрянное произведение","скалярное произвдение"]){
  const lex=index.search(q,{cohort:"x",subject:"math"});
  assert.equal(lex[0]?.row.id,"scalar",q);
  assert.ok(lex.every(h=>h.row.cohort==="x"));
  assert.equal(hybridRank(lex,[{row:rows[1],score:1},{row:rows[0],score:0}])[0].row.id,"scalar");
 }
 assert.equal(index.search("zzzzzzzz",{cohort:"x"}).length,0);
});
test("numeric tokens and ambiguity",()=>{
 assert.deepEqual(lexicalTokens("не 2025 \\neq 2026"),["не","2025","≠","2026"]);
 assert.ok(!near("2025","2036"));
 const ambiguous=new LexicalIndex([
  {id:"a",cohort:"x",subject:"s",title:"",text:"alpha"},
  {id:"b",cohort:"x",subject:"s",title:"",text:"alphi"}
 ]);
 assert.equal(ambiguous.search("alphx",{cohort:"x"}).length,0);
 // If there is no lexical match, label semantic candidates explicitly.
 assert.equal(hybridRank([],[{row:rows[0],score:.5}])[0].kind,"semantic-only");
});

test("real corpus typo benchmark",()=>{
 const corpus=JSON.parse(readFileSync(new URL("../../_site/ai-search-assets/passages.json",import.meta.url),"utf8"));
 const index=new LexicalIndex(corpus);
 const cases=[
  {subject:"analytical_geometry",clean:"скалярное произведение",typo:"сколярное произведение",file:"05 - Скалярное произведение.md",anchor:"\\vec a\\cdot\\vec b="},
  {subject:"analytical_geometry",clean:"скалярное произведение",typo:"скалрянное произвдение",file:"05 - Скалярное произведение.md",anchor:"\\vec a\\cdot\\vec b="},
  {subject:"mathematical_analysis",clean:"пересечение множеств",typo:"пересичение множеств",file:"02 - Операции над множествами и бинарные отношения.md",anchor:"Если 2 множества не имеют общих элементов"},
  {subject:"mathematical_analysis",clean:"объединение множеств",typo:"обьединение множеств",file:"02 - Операции над множествами и бинарные отношения.md",anchor:"C=A\\cup B"},
  {subject:"russian_statehood_fundamentals",clean:"легитимность власти",typo:"легетимность власти",file:"06 - Природа политической власти в РФ.md",anchor:"общественным признанием права власти"},
  {subject:"russian_statehood_fundamentals",clean:"суверенитет",typo:"суверинитет",file:"06 - Природа политической власти в РФ.md",anchor:"верховенство и независимость государственной власти"}
 ];
 const report=cases.map(c=>{
  const matches=r=>r.source.endsWith("/"+c.file)&&r.text.includes(c.anchor);
  const filters={cohort:"ib-2026-1kurs",subject:c.subject,limit:5};
  const expected=corpus.some(r=>matches(r)&&r.cohort===filters.cohort);
  const rank=q=>index.search(q,filters).findIndex(h=>matches(h.row))+1;
  return {query:c.typo,expected,clean_rank:rank(c.clean),typo_rank:rank(c.typo)};
 });
 console.log("AI_TYPO_BENCHMARK="+JSON.stringify(report));
 assert.ok(report.some(x=>x.expected),"source anchors missing");
});
