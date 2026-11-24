import json
from pathlib import Path
import tempfile
import unittest

from ai.indexing.build_index import build
from ai.indexing.chunker import chunk_document
from ai.search.bm25 import BM25
from ai.search.lexical import tokens, near

def row(name, text, cohort=""):
    path = (f"cohorts/{cohort}/" if cohort else "")+"subjects/physics/notes/"+name+".md"
    return chunk_document(path,text)

class SearchTests(unittest.TestCase):
    def setUp(self):
        self.rows = (row("Скорость","# Скорость\n\nСкорость тела — изменение положения.\n")
                     +row("Ускорение","# Ускорение\n\nУскорение тела — изменение скорости.\n")
                     +row("Ускорение","# Ускорение\n\nОсобый ответ другого курса.\n","ib-2025-1kurs"))
        self.engine = BM25(self.rows)

    def test_inflections(self):
        self.assertEqual(tokens("ускорением"),tokens("ускорение"))
        self.assertEqual(tokens("векторами"),tokens("вектор"))
        self.assertEqual(tokens("Что такое?"),[])
        self.assertTrue(near("ускорен","ускроен"))

    def test_relevance_and_verbatim(self):
        hits = self.engine.search("изменение скорости тела ускорением")
        self.assertIn("изменение скорости",hits[0].passage["text"])
        self.assertEqual(hits[0].passage,self.rows[1])

    def test_typo(self):
        hits = self.engine.search("ускроение")
        self.assertEqual(hits[0].passage,self.rows[1])
        self.assertTrue(hits[0].corrected)
        self.assertEqual(self.engine.search("ускроение",fuzzy=False),[])

    def test_filters_and_abstention(self):
        self.assertEqual(self.engine.search("ускорение",subject="obzh"),[])
        self.assertEqual(self.engine.search("ускорение",layers={"exam"}),[])
        self.assertEqual(self.engine.search("Что такое?"),[])
        self.assertEqual(self.engine.search("бульбулятор фырчеплюх"),[])
        self.assertEqual(self.engine.search("ускорение",cohort="ib-2025-1kurs")[0].passage,self.rows[2])
        self.assertTrue(all(h.passage["cohort"]=="ib-2026-1kurs" for h in self.engine.search("ускорение")))

    def test_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root/"subjects/physics/notes/01.md"
            p.parent.mkdir(parents=True)
            p.write_text("# Тема\n\nУскорение.\n",encoding="utf-8")
            out = root/"index"
            build(root,out)
            self.assertTrue(BM25.load(out).search("ускорение"))
            with (out/"passages.jsonl").open("a",encoding="utf-8") as f:
                f.write("\n")
            with self.assertRaises(ValueError):
                BM25.load(out)

if __name__=="__main__":
    unittest.main()
