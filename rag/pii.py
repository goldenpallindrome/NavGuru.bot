from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
_analyzer = None
_anonymizer = AnonymizerEngine()

def engines():
    global _analyzer
    if _analyzer is None:
        _analyzer = AnalyzerEngine()
    return _analyzer

def mask_pii(text: str):
    # Preserve document structure while masking common personal identifiers.
    results = engines().analyze(
        text=text, language="en",
        entities=["PERSON","PHONE_NUMBER","EMAIL_ADDRESS","US_SSN","CREDIT_CARD",
                  "IBAN_CODE","IP_ADDRESS","LOCATION","DATE_TIME"]
    )
    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text, len(results)
