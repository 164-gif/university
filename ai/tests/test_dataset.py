from pathlib import Path
import tempfile
import unittest

from ai.indexing.chunker import chunk_document
from ai.indexing.build_index import build
from ai.tokenizer.bpe import train
from ai.tokenizer.train import run as train_tokenizer
from ai.training.dataset import build_rows, validate, split_passage, make_queries
from ai.training.build_dataset import run

def row(subject,name,text):
    return chunk_document(f"subjects/{subject}/notes/{name}.md",text)

class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.rows=(row("physics","Ускорение","# Ускорение\n\nУскорение — изменение скорости по времени.\n")
                   +row("physics","Скорость","# Скорость\n\nСкорость — изменение положения по времени.\n")
                   +row("mathematical_analysis","Множества","# Множества\n\nМножество — совокупность объектов.\n")
                   +row("russian_statehood_fundamentals","Государство","# Государство\n\nГосударство — организация публичной власти.\n"))
        self.tok=train([r["text"] for r in self.rows],vocab_size=350)

    def test_splits_and_references(self):
        result,stats=build_rows(self.rows,self.tok)
        validate(result,self.tok,512)
        for s,(rows,examples) in result.items():
            self.assertTrue(rows)
            self.assertTrue(examples)
            ids={r["id"] for r in rows}
            for e in examples:
                self.assertTrue(set(e["positive_ids"])<=ids)
                self.assertFalse(set(e["positive_ids"]) & {n["id"] for n in e["negative_candidates"]})
        self.assertEqual(result,build_rows(self.rows,self.tok)[0])

    def test_cross_split_duplicates(self):
        text="# Общая тема\n\nТермин — одинаковый достаточно длинный текст.\n"
        rows=self.rows+row("physics","a",text)+row("mathematical_analysis","b",text)
        result,stats=build_rows(rows,self.tok)
        validate(result,self.tok,512)
        self.assertEqual(stats["excluded_cross_split_duplicate"],2)

    def test_qa_and_qualification(self):
        r=row("physics","Тест","# Тест\n\n**Вопрос 1. Что такое ускорение?**\n\n**Ответ:** изменение скорости.\n")[0]
        queries=dict(make_queries(r,False))
        self.assertIn("Что такое ускорение?",queries)
        self.assertIn("Дайте определение: ускорение",queries)
        self.assertEqual(list(split_passage(r,self.tok,8)),[])

    def test_long_source_exact_slices(self):
        src="# Тема\n\nПервый абзац текста.\n\n$$\nx+y=z\n$$\n\n"+"я"*500+"\n\nДругой текст.\n"
        r=row("physics","long",src)[0]
        parts=list(split_passage(r,self.tok,100))
        self.assertTrue(parts)
        for part in parts:
            self.assertEqual(part["text"],"".join(src.splitlines(keepends=True)[part["line_start"]-1:part["line_end"]]))
            self.assertLessEqual(len(self.tok.encode(" / ".join(part["headings"])+"\n\n"+part["text"])),100)
            if "$$" in part["text"]:
                self.assertEqual(part["text"].count("$$"),2)

    def test_pipeline_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for r in self.rows:
                p=root/r["source"]
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text("# "+r["headings"][0]+"\n\n"+r["text"],encoding="utf-8")
            out=root/"generated"
            build(root,out)
            train_tokenizer(out,out,350)
            report=run(out,out/"dataset")
            self.assertTrue(all(s["examples"] for s in report["splits"].values()))
            self.assertEqual(report,run(out,out/"dataset"))
            (out/"tokenizer.json").write_text("{}",encoding="utf-8")
            with self.assertRaises(ValueError):
                run(out,out/"dataset")

if __name__=="__main__":
    unittest.main()
