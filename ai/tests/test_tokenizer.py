import json
from pathlib import Path
import random
import tempfile
import unittest

from ai.tokenizer.bpe import BPETokenizer, BASE_SIZE, PAD, BOS, EOS, split, train
from ai.tokenizer.train import run
from ai.indexing.build_index import build

class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.texts=["Скалярное произведение векторов.\n", "Векторное произведение векторов.",
                    r"\vec a\cdot\vec b=\cos\alpha"]*10
        self.tok=train(self.texts,vocab_size=340)

    def test_roundtrip_unknown_unicode(self):
        for t in ["", "Ёж 🧪 λ ∀x∈ℝ", "e\u0301 ≠ é", " \t\r\n",
                  r"\frac{a_1}{b^2}", "[PAD] [BOS]", "\x00", "𐀀"]:
            self.assertEqual("".join(split(t)),t)
            self.assertEqual(self.tok.decode(self.tok.encode(t)),t)
        rng=random.Random(42)
        for _ in range(40):
            t="".join(chr(rng.choice([rng.randrange(0,0xd800),rng.randrange(0xe000,0x110000)])) for _ in range(30))
            self.assertEqual(self.tok.decode(self.tok.encode(t)),t)

    def test_deterministic_and_compression(self):
        self.assertEqual(self.tok.merges,train(reversed(self.texts),vocab_size=340).merges)
        self.assertLess(len(self.tok.encode(self.texts[0],add_special=False)),len(self.texts[0].encode()))
        self.assertLessEqual(self.tok.vocab_size,340)

    def test_overlapping_pair_and_ties(self):
        t=train(["aaaa","bbbb"],vocab_size=BASE_SIZE+1)
        self.assertEqual(t.merges,((ord("a")+4,ord("a")+4),))
        self.assertEqual(len(t.encode("aaaa",add_special=False)),2)
        self.assertEqual(t.decode(t.encode("aaaaa")),"aaaaa")

    def test_model_inputs(self):
        x=self.tok.model_inputs("λ",max_length=8)
        self.assertEqual(len(x["input_ids"]),8)
        self.assertEqual(x["input_ids"][0],BOS)
        self.assertEqual(x["input_ids"][sum(x["attention_mask"])-1],EOS)
        self.assertTrue(all(i==PAD for i in x["input_ids"][sum(x["attention_mask"]):]))
        self.assertFalse(x["truncated"])
        with self.assertRaises(ValueError):
            self.tok.model_inputs("abcdefgh",max_length=2)
        self.assertTrue(self.tok.model_inputs("abcdefgh",max_length=2,truncate=True)["truncated"])

    def test_serialization(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"tokenizer.json"
            self.tok.save(p)
            self.assertEqual(BPETokenizer.load(p).encode(self.texts[0]),self.tok.encode(self.texts[0]))
            data=json.loads(p.read_text())
            data["merges"][0]=[0,1]
            p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                BPETokenizer.load(p)

    def test_training_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=root/"subjects/physics/notes/01.md"
            p.parent.mkdir(parents=True)
            p.write_text("# Ускорение\n\nУскорение ускорение.\n",encoding="utf-8")
            out=root/"generated"
            build(root,out)
            report=run(out,out,300)
            self.assertEqual(report["unique_passages"],report["roundtrip_passages"])
            self.assertTrue((out/"tokenizer.json").exists())
            with (out/"passages.jsonl").open("a") as f:
                f.write("\n")
            with self.assertRaises(ValueError):
                run(out,out,300)

if __name__=="__main__":
    unittest.main()
