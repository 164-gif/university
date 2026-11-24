import random
import unittest
from ai.question_bank.typos import augment,corrupt,norm
from ai.question_bank.build import scan_questions

class BankTests(unittest.TestCase):
    def test_noisy_preserves_metadata(self):
        row={"id":"abc","query":"Скалярное произведение векторов $a=10$ не равно 20?","positive_ids":["answer"],"split":"test","label":"source_answer"}
        a=augment([row],5,42)
        self.assertEqual(a,augment([row],5,42))
        self.assertEqual(len(a),5)
        self.assertEqual(len({norm(x["query"]) for x in a}),5)
        for e in a:
            self.assertIn("$a=10$ не равно 20",e["query"])
            self.assertEqual(e["positive_ids"],["answer"])
            self.assertEqual(e["split"],"test")
            self.assertEqual(e["parent_id"],"abc")
    def test_review_not_augmented(self):
        self.assertEqual(augment([{"id":"x","query":"Произведение векторов","positive_ids":[],"label":"review"}]),[])
    def test_protect_formula(self):
        s="Что значит $\\mathrm{скорость}=123$ и не равно 42?"
        self.assertEqual(corrupt(s,"delete",random.Random(1))[0],s)
    def test_question_without_answer(self):
        r=list(scan_questions("subjects/physics/exam/QUESTIONS.md",("c","physics","exam"),"## Вопросы\n\n1. Что такое масса?\n2. Как найти силу?\n",[]))
        self.assertEqual(len(r),2)
        self.assertTrue(all(x["label"]=="review" for x in r))

if __name__=="__main__":
    unittest.main()
