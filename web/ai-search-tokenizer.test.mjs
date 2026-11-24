import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { UniversityBPETokenizer } from "./ai-search-tokenizer.js";

const tokenizerPath = fileURLToPath(new URL("../ai/generated/full/tokenizer.json", import.meta.url));

test("byte-BPE special tokens and byte offsets", () => {
  const tok = new UniversityBPETokenizer({version:1,algorithm:"byte-bpe",special:["[PAD]","[BOS]","[EOS]","[MASK]"],vocab_size:261,merges:[[101,102]]});
  assert.deepEqual(tok.encode("ab"), [1,260,2]);
  assert.deepEqual(tok.encode("a🙂"), [1,101,244,163,157,134,2]);
});

test("browser tokenizer matches Python for Unicode, math and punctuation", {skip:!existsSync(tokenizerPath)}, () => {
  const samples=["Что такое множество?","Вероятность — мера возможности события.","\\vec a\\cdot\\vec b=|a||b|\\cos\\alpha","x₁ + 2 = 10\nТекст 🙂"];
  const json=JSON.parse(readFileSync(tokenizerPath,"utf8"));
  const tok=new UniversityBPETokenizer(json);
  const actual=samples.map(text=>tok.encode(text));
  const expected=[
    [1,578,36,819,36,10826,67,2],
    [1,9240,36,335,36,2564,36,3243,36,7836,50,2],
    [1,415,36,101,765,415,36,102,7108,101,128,128,102,522,696,805,2],
    [1,124,230,134,133,36,47,36,54,36,65,36,554,14,438,348,279,36,244,163,157,134,2]
  ];
  assert.deepEqual(actual,expected);
});
