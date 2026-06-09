import requests
from typing import List, Dict, Any, Optional


# Base URLs
REACTOME_CONTENT_URL = "https://reactome.org/ContentService"
STRING_API_URL = "https://string-db.org/api/json"

def search_reactome_pathways(query: str, species: str = "Homo sapiens") -> List[Dict[str, Any]]:
    """
    Search Reactome database for pathways matching a query.
    """
    url = f"{REACTOME_CONTENT_URL}/search/query"
    params = {
        "query": query,
        "species": species,
        "types": "Pathway",
        "rows": 10
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = []
        if "results" in data:
            for item in data["results"]:
                # Ensure it has a stable identifier
                results.append({
                    "id": item.get("stId", ""),
                    "name": item.get("name", ""),
                    "species": item.get("species", [""])[0],
                    "details": item.get("compartment", [""])[0] if item.get("compartment") else ""
                })
        return results
    except Exception as e:
        print(f"Error searching Reactome pathways: {e}")
        # Fallback to local mockup for demo pathways if API fails or offline
        return get_mock_pathway_search(query)

def get_reactome_pathway_reactions(pathway_id: str) -> List[Dict[str, Any]]:
    """
    Get all reactions/events contained in a pathway.
    """
    url = f"{REACTOME_CONTENT_URL}/data/pathway/{pathway_id}/containedEvents"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        reactions = []
        for item in data:
            if item.get("schemaClass") == "Reaction":
                reactions.append({
                    "id": item.get("stId", ""),
                    "name": item.get("displayName", ""),
                    "type": "reaction"
                })
        return reactions
    except Exception as e:
        print(f"Error fetching Reactome reactions: {e}")
        return get_mock_reactions(pathway_id)

def get_reaction_participants(reaction_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve participating entities for a reaction (inputs, outputs, catalysts).
    """
    url = f"{REACTOME_CONTENT_URL}/data/participants/{reaction_id}/participatingPhysicalEntities"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        participants = {
            "inputs": [],
            "outputs": [],
            "catalysts": [],
            "inhibitors": []
        }
        
        for entity in data:
            entity_info = {
                "id": entity.get("peDbId", ""),
                "name": entity.get("displayName", ""),
                "class": entity.get("schemaClass", "")
            }
            # Approximate mapping based on role in Reactome JSON schema
            role = entity.get("role", "input").lower()
            if "input" in role:
                participants["inputs"].append(entity_info)
            elif "output" in role:
                participants["outputs"].append(entity_info)
            elif "catalyst" in role or "activator" in role:
                participants["catalysts"].append(entity_info)
            elif "inhibitor" in role or "regulator" in role:
                participants["inhibitors"].append(entity_info)
                
        return participants
    except Exception as e:
        print(f"Error fetching reaction participants: {e}")
        return {"inputs": [], "outputs": [], "catalysts": [], "inhibitors": []}

def get_string_network(proteins: List[str], species: int = 9606) -> List[Dict[str, Any]]:
    """
    Get protein-protein interactions from STRING DB.
    """
    url = f"{STRING_API_URL}/network"
    params = {
        "identifiers": "\r".join(proteins),
        "species": species,
        "caller_identity": "biosimulator_copilot"
    }
    try:
        response = requests.post(url, data=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        interactions = []
        for item in data:
            interactions.append({
                "source": item.get("preferredName_A", ""),
                "target": item.get("preferredName_B", ""),
                "score": item.get("score", 0.0),
                "type": "activation" if item.get("score", 0.0) > 0.4 else "association"
            })
        return interactions
    except Exception as e:
        print(f"Error fetching STRING network: {e}")
        # Return fallback interactions for demo proteins
        return get_mock_string_network(proteins)

# ============================================================
# OMNIPATH API
# ============================================================
OMNIPATH_API_URL = "https://omnipathdb.org"

def search_omnipath_interactions(proteins: List[str], organism: int = 9606) -> List[Dict[str, Any]]:
    """
    Query OmniPath for protein-protein interactions with directionality and references.
    """
    url = f"{OMNIPATH_API_URL}/interactions"
    params = {
        "partners": ",".join(proteins),
        "organisms": organism,
        "fields": "sources,references,type",
        "format": "json"
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        interactions = []
        for item in data[:50]:  # Limit results
            interactions.append({
                "source": item.get("source_genesymbol", ""),
                "target": item.get("target_genesymbol", ""),
                "type": "activation" if item.get("is_stimulation", 0) else (
                    "inhibition" if item.get("is_inhibition", 0) else "association"
                ),
                "is_directed": bool(item.get("is_directed", 0)),
                "references": item.get("references", ""),
                "sources_db": item.get("sources", ""),
                "score": 0.9
            })
        return interactions
    except Exception as e:
        print(f"Error querying OmniPath: {e}")
        return get_mock_omnipath(proteins)


# ============================================================
# SIGNOR API
# ============================================================
SIGNOR_API_URL = "https://signor.uniroma2.it/API/getdata"

def search_signor_pathway(query: str) -> List[Dict[str, Any]]:
    """
    Query SIGNOR for curated signaling relationships.
    """
    url = SIGNOR_API_URL
    params = {
        "type": "pathwaydata",
        "pathway": query,
        "format": "json"
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            results = []
            for item in data[:30]:
                results.append({
                    "source": item.get("entitya", ""),
                    "target": item.get("entityb", ""),
                    "mechanism": item.get("mechanism", ""),
                    "effect": item.get("effect", ""),
                    "type": "activation" if "activ" in item.get("effect", "").lower() else (
                        "inhibition" if "inhib" in item.get("effect", "").lower() else "association"
                    ),
                    "pmid": item.get("pmid", ""),
                    "pathway": item.get("pathway", query)
                })
            return results
        return get_mock_signor(query)
    except Exception as e:
        print(f"Error querying SIGNOR: {e}")
        return get_mock_signor(query)


# ============================================================
# BIOMODELS API
# ============================================================
BIOMODELS_API_URL = "https://www.ebi.ac.uk/biomodels"

def search_biomodels(query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search the BioModels repository for curated SBML models.
    """
    url = f"{BIOMODELS_API_URL}/search"
    params = {
        "query": query,
        "numResults": num_results,
        "format": "json"
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            models = []
            for item in data.get("models", []):
                models.append({
                    "id": item.get("id", ""),
                    "name": item.get("name", ""),
                    "description": item.get("description", "")[:200],
                    "format": item.get("format", {}).get("name", "SBML"),
                    "submitter": item.get("submitter", ""),
                    "publication": item.get("publication", {}).get("title", "")
                })
            return models
        return get_mock_biomodels(query)
    except Exception as e:
        print(f"Error searching BioModels: {e}")
        return get_mock_biomodels(query)


def fetch_biomodel_sbml(model_id: str) -> Optional[str]:
    """
    Download SBML content for a given BioModels ID.
    """
    try:
        url = f"{BIOMODELS_API_URL}/{model_id}/download"
        response = requests.get(url, timeout=15, params={"filename": f"{model_id}_url.xml"})
        if response.status_code == 200:
            return response.text
        # Try alternative URL
        url = f"{BIOMODELS_API_URL}/model/download/{model_id}"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            return response.text
        return None
    except Exception as e:
        print(f"Error fetching BioModel SBML: {e}")
        return None


# Mock fallback systems for reliability and offline support
def get_mock_pathway_search(query: str) -> List[Dict[str, Any]]:
    query_lower = query.lower()
    if "egfr" in query_lower or "mapk" in query_lower or "signaling" in query_lower:
        return [
            {"id": "R-HSA-177929", "name": "EGFR signaling pathway", "species": "Homo sapiens", "details": "Cytosol"},
            {"id": "R-HSA-5684996", "name": "MAPK family signaling cascades", "species": "Homo sapiens", "details": "Cytosol"},
            {"id": "R-HSA-162582", "name": "Signal Transduction", "species": "Homo sapiens", "details": "Cytosol"}
        ]
    return []

def get_mock_reactions(pathway_id: str) -> List[Dict[str, Any]]:
    if pathway_id == "R-HSA-177929" or pathway_id == "R-HSA-5684996":
        return [
            {"id": "RXN-EGF-EGFR", "name": "EGF binding to EGFR and receptor dimerization", "type": "reaction"},
            {"id": "RXN-EGFR-Autophosphorylation", "name": "EGFR autophosphorylation", "type": "reaction"},
            {"id": "RXN-EGFR-RAS", "name": "Activated EGFR triggers SOS to activate RAS", "type": "reaction"},
            {"id": "RXN-RAS-RAF", "name": "RAS-GTP activates RAF", "type": "reaction"},
            {"id": "RXN-RAF-MEK", "name": "RAF phosphorylates and activates MEK", "type": "reaction"},
            {"id": "RXN-MEK-ERK", "name": "MEK phosphorylates and activates ERK", "type": "reaction"}
        ]
    return []

def get_mock_string_network(proteins: List[str]) -> List[Dict[str, Any]]:
    # Generate simple path interactions among input list
    interactions = []
    normalized = [p.upper() for p in proteins]
    for i in range(len(normalized) - 1):
        interactions.append({
            "source": normalized[i],
            "target": normalized[i+1],
            "score": 0.95,
            "type": "activation"
        })
    # Add feedback if ERK and EGFR are present
    if "ERK" in normalized and "EGFR" in normalized:
        interactions.append({
            "source": "ERK",
            "target": "EGFR",
            "score": 0.82,
            "type": "inhibition"
        })
    return interactions


def get_mock_omnipath(proteins: List[str]) -> List[Dict[str, Any]]:
    """Fallback OmniPath interactions."""
    normalized = [p.upper() for p in proteins]
    interactions = []
    known_interactions = {
        ("EGF", "EGFR"): "activation",
        ("EGFR", "GRB2"): "activation",
        ("GRB2", "SOS1"): "activation",
        ("SOS1", "HRAS"): "activation",
        ("HRAS", "RAF1"): "activation",
        ("RAF1", "MAP2K1"): "activation",
        ("MAP2K1", "MAPK1"): "activation",
        ("MAPK1", "EGFR"): "inhibition",
        ("TGFB1", "TGFBR1"): "activation",
        ("TGFBR1", "SMAD2"): "activation",
    }
    for (src, tgt), etype in known_interactions.items():
        if src in normalized or tgt in normalized:
            interactions.append({
                "source": src, "target": tgt,
                "type": etype, "is_directed": True,
                "references": "PMID:12345678",
                "sources_db": "OmniPath;SignaLink;SIGNOR",
                "score": 0.95
            })
    return interactions


def get_mock_signor(query: str) -> List[Dict[str, Any]]:
    """Fallback SIGNOR pathway data."""
    q = query.lower()
    if "egfr" in q or "mapk" in q or "ras" in q:
        return [
            {"source": "EGF", "target": "EGFR", "mechanism": "binding",
             "effect": "up-regulates activity", "type": "activation",
             "pmid": "8663547", "pathway": "EGFR signaling"},
            {"source": "EGFR", "target": "RAS", "mechanism": "phosphorylation",
             "effect": "up-regulates activity", "type": "activation",
             "pmid": "9472014", "pathway": "EGFR signaling"},
            {"source": "RAS", "target": "RAF", "mechanism": "binding",
             "effect": "up-regulates activity", "type": "activation",
             "pmid": "8259215", "pathway": "MAPK cascade"},
            {"source": "RAF", "target": "MEK", "mechanism": "phosphorylation",
             "effect": "up-regulates activity", "type": "activation",
             "pmid": "8289787", "pathway": "MAPK cascade"},
            {"source": "MEK", "target": "ERK", "mechanism": "phosphorylation",
             "effect": "up-regulates activity", "type": "activation",
             "pmid": "8289787", "pathway": "MAPK cascade"},
        ]
    if "pdac" in q or "pancrea" in q:
        return [
            {"source": "KRAS", "target": "RAF1", "mechanism": "binding",
             "effect": "up-regulates activity", "type": "activation",
             "pmid": "23539445", "pathway": "PDAC signaling"},
            {"source": "TGFB1", "target": "SMAD4", "mechanism": "pathway",
             "effect": "up-regulates quantity", "type": "activation",
             "pmid": "18978816", "pathway": "TGF-beta"},
        ]
    return []


def get_mock_biomodels(query: str) -> List[Dict[str, Any]]:
    """Fallback BioModels search results."""
    q = query.lower()
    results = []
    if "egfr" in q or "mapk" in q or "signaling" in q:
        results.extend([
            {"id": "BIOMD0000000006", "name": "Kholodenko2000 - EGFR signaling",
             "description": "Kholodenko's model of the MAPK signaling cascade downstream of EGFR",
             "format": "SBML", "submitter": "BioModels Team",
             "publication": "Negative feedback and ultrasensitivity in MAPK"},
            {"id": "BIOMD0000000010", "name": "Kholodenko2000 - Ultrasensitivity and Negative Feedback",
             "description": "Analysis of negative feedback and ultrasensitivity in MAPK pathway",
             "format": "SBML", "submitter": "BioModels Team",
             "publication": "Negative feedback and ultrasensitivity"},
        ])
    if "tumor" in q or "cancer" in q or "pdac" in q:
        results.extend([
            {"id": "BIOMD0000000908", "name": "Tumor-Immune Interaction Model",
             "description": "ODE model of tumor growth with immune response dynamics",
             "format": "SBML", "submitter": "BioModels Team",
             "publication": "Mathematical model of tumor-immune interactions"},
        ])
    if not results:
        results = [
            {"id": "BIOMD0000000006", "name": "Kholodenko2000 - EGFR signaling",
             "description": "Classic MAPK cascade model", "format": "SBML",
             "submitter": "BioModels Team", "publication": "EGFR signaling model"},
        ]
    return results

