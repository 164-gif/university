import json
from pathlib import Path
import tempfile
import unittest

from ai.indexing.chunker import classify, chunk_document, digest
from ai.indexing.build_index import build, ROOT

NOTE = "subjects/physics/notes/01 - Тест.md"
QA = "subjects/mathematical_analysis/exam/Тест.md"

class ChunkTests(unittest.TestCase):
    def test_verbatim_math_and_frontmatter(self):
        source = '---\r\nstatus: draft\r\n---\r\n# Физика\r\n\r\n## Ускорение\r\n\r\nОпределение.\r\n\r\n$$\r\n\\vec a=\\frac{d\\vec v}{dt}\r\n\r\n+0\r\n$$\r\n\r\nДругой текст.\r\n'
        rows = chunk_document(NOTE, source, target=12)
        self.assertIn("Определение.", rows[0]["text"])
        self.assertIn("$$\r\n\\vec a", rows[0]["text"])
        self.assertIn("+0\r\n$$", rows[0]["text"])
        self.assertTrue(rows[0]["document_warning"])
        for row in rows:
            self.assertEqual(row["text"], "".join(source.splitlines(keepends=True)[row["line_start"]-1:row["line_end"]]))
            self.assertEqual(row["text_sha256"], digest(row["text"]))
            self.assertEqual(row["headings"], ["Физика", "Ускорение"])

    def test_question_answer_not_split(self):
        text = "# Тест\n\n**Вопрос 1. Что такое множество?**\n\n**Ответ:** определение.\n\nОговорка: не официальный ключ.\n\n**2. Что такое пустое множество?**\nОтвет: множество без элементов.\n"
        rows = chunk_document(QA, text, target=15)
        self.assertEqual(len(rows), 2)
        self.assertIn("Оговорка", rows[0]["text"])
        self.assertIn("Ответ", rows[1]["text"])
        self.assertTrue(rows[0]["passage_warning"])

    def test_fenced_heading_not_structural(self):
        text = "# Тема\n\n~~~python\n# comment\n\nprint(1)\n~~~\n\n## Далее\nТекст\n"
        rows = chunk_document(NOTE, text, 10)
        self.assertIn("# comment\n\nprint(1)", rows[0]["text"])
        self.assertEqual(rows[0]["headings"], ["Тема"])
        self.assertEqual(rows[-1]["headings"], ["Тема", "Далее"])

    def test_layers_and_cohorts(self):
        self.assertEqual(classify("cohorts/ib-2025-1kurs/subjects/Осипов/exam/Вопросы.md"),
                         ("ib-2025-1kurs", "Осипов", "exam"))
        self.assertEqual(classify("subjects/laboratory_physics/lab_02/teacher_questions.md")[-1],
                         "teacher_questions")
        for path in ("subjects/physics/raw/a.md", "subjects/physics/checks/a.md",
                     "subjects/physics/notes/assets/a.md", "subjects/physics/README.md"):
            self.assertIsNone(classify(path))

    def test_deterministic_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / NOTE
            p.parent.mkdir(parents=True)
            p.write_bytes("# Тема\n\nТекст.\n".encode())
            out = root / "generated"
            first = build(root, out)
            data = (out / "passages.jsonl").read_bytes()
            second = build(root, out)
            self.assertEqual(first, second)
            self.assertEqual(data, (out / "passages.jsonl").read_bytes())
            self.assertEqual(first["index_sha256"], digest(data.decode()))
            self.assertEqual(json.loads(data)["text"], "Текст.\n")
            self.assertEqual(p.read_bytes(), "# Тема\n\nТекст.\n".encode())
            with self.assertRaises(ValueError):
                build(root, p.parent)

    def test_real_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = build(ROOT, out)
            self.assertGreater(manifest["documents"], 0)
            self.assertGreater(manifest["passages"], manifest["documents"])
            rows = [json.loads(line) for line in (out / "passages.jsonl").read_text(encoding="utf-8").splitlines()]
            for row in rows:
                source = (ROOT / row["source"]).read_bytes().decode("utf-8")
                self.assertEqual(row["text"], "".join(source.splitlines(keepends=True)[row["line_start"]-1:row["line_end"]]))
                self.assertEqual(row["source_sha256"], digest(source))
            scalar = [r for r in rows if r["source"].endswith("05 - Скалярное произведение.md") and r["layer"] == "notes"]
            self.assertTrue(any(r"\vec a\cdot\vec b" in r["text"] for r in scalar))
            print(f"Corpus: {manifest['documents']} documents, {manifest['passages']} passages")

if __name__ == "__main__":
    unittest.main()
