from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_abm_preset():
    print("Testing ABM preset endpoint...")
    response = client.get("/api/abm/preset/tumor_growth")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert data["name"] == "Tumor Growth (Nutrient-Dependent)"
    assert "cell_types" in data
    assert len(data["cell_types"]) >= 1
    print("[OK] ABM Preset 'tumor_growth' successfully loaded.")

if __name__ == "__main__":
    test_abm_preset()
