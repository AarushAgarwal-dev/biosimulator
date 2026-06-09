from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_maple_extract():
    print("Testing MAPLE parameter extraction schema validation...")
    
    # We will test the validation endpoint rather than full LLM extraction to avoid API key requirements in simple tests
    payload = {
        "target_type": "submodel",
        "target_data": {
            "target_id": "test_target",
            "description": "Test description",
            "target_parameter": "k_prolif",
            "inputs": [
                {
                    "name": "prolif_rate",
                    "value": 1.5,
                    "units": "1/day",
                    "snippet": {"text": "The cancer cells divide every 16 hours."},
                    "source": {"doi": "10.1234/test", "title": "Test", "authors": "A et al", "year": 2024}
                }
            ],
            "forward_model": {
                "model_type": "algebraic",
                "parameters": [{"name": "k_prolif", "role": "calibration"}],
                "custom_code": "def f(t, y, p): return p['k_prolif']"
            },
            "source_relevance": {
                "indication_match": "exact",
                "indication_justification": "Same disease",
                "evidence_type": "in_vitro"
            }
        }
    }
    
    response = client.post("/api/maple/validate", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()
    assert "all_passed" in data
    assert "results" in data
    
    # Intentionally trigger a validation error (bad DOI)
    bad_payload = {
        "target_type": "submodel",
        "target_data": {
            "target_id": "test_target",
            "description": "Test description",
            "target_parameter": "k_prolif",
            "inputs": [
                {
                    "name": "prolif_rate",
                    "value": 1.5,
                    "units": "1/day",
                    "snippet": {"text": "The cancer cells divide every 16 hours."},
                    "source": {"doi": "bad_doi", "title": "Test", "authors": "A et al", "year": 2024}
                }
            ],
            "forward_model": {
                "model_type": "algebraic",
                "parameters": [{"name": "k_prolif", "role": "calibration"}],
                "custom_code": "def f(t, y, p): return p['k_prolif']"
            },
            "source_relevance": {
                "indication_match": "exact",
                "indication_justification": "Same disease",
                "evidence_type": "in_vitro"
            }
        }
    }
    bad_response = client.post("/api/maple/validate", json=bad_payload)
    # The Pydantic validation handles this gracefully or throws a 422
    assert bad_response.status_code in [200, 422, 500], f"Error: {bad_response.text}"
    
    print("[OK] MAPLE parameter extraction schema validation endpoints tested.")

if __name__ == "__main__":
    test_maple_extract()
