import unittest
from rag.query_understanding import classify_query, assess_corpus_compatibility

class QueryUnderstandingTests(unittest.TestCase):
    def test_medical_query_rejected_by_finance_only_corpus(self):
        result=classify_query("What are the symptoms of diabetes?")
        gate=assess_corpus_compatibility(result,{"finance":100})
        self.assertEqual(result["domain"],"medical")
        self.assertEqual(gate["status"],"domain_mismatch")

    def test_finance_query_allowed_by_finance_corpus(self):
        result=classify_query("Explain quarterly revenue and cash flow")
        gate=assess_corpus_compatibility(result,{"finance":100})
        self.assertEqual(result["domain"],"finance")
        self.assertEqual(gate["status"],"proceed")

    def test_unknown_query_is_not_rejected(self):
        result=classify_query("Summarize this document")
        gate=assess_corpus_compatibility(result,{"finance":100})
        self.assertEqual(gate["status"],"proceed")

if __name__ == "__main__":
    unittest.main()
