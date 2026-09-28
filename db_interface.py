import re
import time
import requests
from typing import List, Dict, Any, Optional


def _strip_html(text: str) -> str:
    """Remove HTML highlighting tags Reactome embeds in search result names."""
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _named_field(value: Any, default: str = "") -> str:
    """Read a field the APIs return inconsistently as a bare string or an object.

    BioModels answers 'format' as the string "SBML" but 'publication' as
    {"title": ...}, so a fixed .get("name") chain raised
    AttributeError: 'str' object has no attribute 'get' and silently dropped every
    live result to the offline set.
    """
    if isinstance(value, dict):
        for key in ("name", "title", "label", "id"):
            nested = value.get(key)
            if nested:
                return str(nested)
        return default
    if value is None or value == "":
        return default
    return str(value)


def _looks_like_sbml(text: str) -> bool:
    """True when a payload is actually SBML XML rather than HTML or a ZIP."""
    return "<sbml" in (text or "")[:4000].lower()


#: BioModels ids are alphanumeric; anything else is refused before it reaches a URL path.
_VALID_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


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
        "rows": 25
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = []
        # Reactome groups results by type: data["results"] -> [{entries: [...]}]
        for group in data.get("results", []):
            for item in group.get("entries", []):
                species_list = item.get("species", [])
                if isinstance(species_list, str):
                    species_list = [species_list]
                results.append({
                    "id": item.get("stId", "") or item.get("id", ""),
                    "name": _strip_html(item.get("name", "")),
                    "species": species_list[0] if species_list else "",
                    "details": item.get("exactType", item.get("type", ""))
                })
        return results if results else get_mock_pathway_search(query)
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
        # containedEvents mixes full event dicts with bare dbId integers for
        # events Reactome did not inline; skip the ints and keep reaction-like events.
        reaction_classes = {
            "Reaction", "ReactionLikeEvent", "BlackBoxEvent",
            "Polymerisation", "Depolymerisation", "FailedReaction"
        }
        for item in data:
            if not isinstance(item, dict):
                continue
            if item.get("schemaClass") in reaction_classes:
                reactions.append({
                    "id": item.get("stId", ""),
                    "name": item.get("displayName", ""),
                    "type": "reaction"
                })
        return reactions if reactions else get_mock_reactions(pathway_id)
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
        # Pull in extra interactors so the network isn't limited to just the
        # typed proteins, and lower the confidence floor to capture more edges.
        "add_nodes": 15,
        "required_score": 200,
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
        # genesymbols=1 adds source_genesymbol/target_genesymbol; without it
        # OmniPath returns only UniProt IDs and the gene-symbol fields are empty.
        "genesymbols": "1",
        "fields": "sources,references,type",
        "format": "json"
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        interactions = []
        for item in data[:300]:  # Capture a broad set of interactions
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
# The '/API/getdata?type=pathwaydata' form this used before answers 404 (checked
# 2026-09-28), so every search silently fell back to the offline examples.
# getData.php is SIGNOR's documented download endpoint; it takes a UniProt accession
# and returns one curated causal relation per tab-separated line.
SIGNOR_DATA_URL = "https://signor.uniroma2.it/getData.php"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
_UNIPROT_ACCESSION = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")
_GENE_SYMBOL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,19}$")
# getData.php columns (0-based): 0 entity A, 4 entity B, 8 effect, 9 mechanism,
# 21 PMID, 26 SIGNOR id, 27 score.
_SIGNOR_COLUMNS = {"a": 0, "b": 4, "effect": 8, "mechanism": 9, "pmid": 21, "id": 26, "score": 27}


def _uniprot_accession(query: str) -> Optional[str]:
    """Map a human gene symbol to its reviewed UniProt accession (an accession passes through)."""
    token = (query or "").strip()
    if _UNIPROT_ACCESSION.match(token.upper()):
        return token.upper()
    if not _GENE_SYMBOL.match(token):
        return None
    response = requests.get(UNIPROT_SEARCH_URL, timeout=10, params={
        "query": f"gene_exact:{token} AND organism_id:9606 AND reviewed:true",
        "fields": "accession", "format": "tsv", "size": 1})
    response.raise_for_status()
    lines = [line for line in response.text.strip().splitlines() if line.strip()]
    return lines[1].split("\t")[0].strip() if len(lines) > 1 else None


def _signor_type(effect: str) -> str:
    effect = (effect or "").strip().lower()
    if effect.startswith("up-regulates"):
        return "activation"
    if effect.startswith("down-regulates"):
        return "inhibition"
    return "association"


def search_signor_pathway(query: str) -> List[Dict[str, Any]]:
    """Curated SIGNOR causal relations involving one human protein (gene symbol or UniProt id)."""
    try:
        accession = _uniprot_accession(query)
        if not accession:
            return []                      # no reviewed human protein by that name
        response = requests.get(SIGNOR_DATA_URL, params={"organism": "9606", "id": accession}, timeout=15)
        response.raise_for_status()
        text = response.text.strip()
        if not text or text.lower().startswith("no result"):
            return []
        results = []
        c = _SIGNOR_COLUMNS
        for line in text.splitlines():
            cols = line.split("\t")
            if len(cols) <= c["mechanism"]:
                continue
            pmid = cols[c["pmid"]].strip() if len(cols) > c["pmid"] else ""
            try:
                score = float(cols[c["score"]]) if len(cols) > c["score"] else None
            except ValueError:
                score = None
            results.append({
                "source": cols[c["a"]].strip(),
                "target": cols[c["b"]].strip(),
                "mechanism": cols[c["mechanism"]].strip(),
                "effect": cols[c["effect"]].strip(),
                "type": _signor_type(cols[c["effect"]]),
                "pmid": pmid if pmid.isdigit() else "",
                "signor_id": cols[c["id"]].strip() if len(cols) > c["id"] else "",
                "score": score,
            })
            if len(results) >= 300:
                break
        return results
    except Exception as e:
        print(f"Error querying SIGNOR: {type(e).__name__}")
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
    if _biomodels_known_down():
        return get_mock_biomodels(query)
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("BioModels search did not return a JSON object.")
            models = []
            for item in data.get("models", []):
                if not isinstance(item, dict):
                    continue
                models.append({
                    "id": str(item.get("id") or ""),
                    "name": str(item.get("name") or ""),
                    "description": str(item.get("description") or "")[:200],
                    "format": _named_field(item.get("format"), "SBML"),
                    "submitter": _named_field(item.get("submitter")),
                    "publication": _named_field(item.get("publication")),
                    "url": str(item.get("url") or ""),
                })
            if models:
                return models
            # An empty parse on a 200 means the schema moved; prefer the offline
            # set over handing the UI an empty list that looks like "no results".
            print("BioModels search returned no parseable models; using offline set.")
        return get_mock_biomodels(query)
    except requests.exceptions.RequestException as e:
        print(f"Error searching BioModels: {type(e).__name__}")
        _mark_biomodels_down()
        return get_mock_biomodels(query)
    except Exception as e:
        print(f"Error searching BioModels: {e}")
        return get_mock_biomodels(query)


#: A CC0 GitHub mirror of the curated BioModels collection (sys-bio/temp-biomodels): 'final'
#: holds the files as re-curated for BioModels, 'original' the files as first published.
#: Used only when BioModels itself does not answer (its host moved to www.biomodels.org and
#: timed out for every request on 2026-09-28), and only for curated BIOMD ids.
BIOMODELS_MIRROR_URLS = (
    "https://raw.githubusercontent.com/sys-bio/temp-biomodels/main/final/{mid}/{mid}_url.xml",
    "https://raw.githubusercontent.com/sys-bio/temp-biomodels/main/original/{mid}/{mid}_url.xml",
)
_CURATED_BIOMD = re.compile(r"^BIOMD\d{10}$")

# When BioModels does not answer at all, remember it briefly so the next search or import
# goes straight to the offline set / mirror instead of waiting out the same timeouts again.
_BIOMODELS_OUTAGE_SECONDS = 600.0
_biomodels_down_until = 0.0


def _biomodels_known_down() -> bool:
    return time.monotonic() < _biomodels_down_until


def _mark_biomodels_down() -> None:
    global _biomodels_down_until
    _biomodels_down_until = time.monotonic() + _BIOMODELS_OUTAGE_SECONDS


def fetch_biomodel_sbml_with_source(model_id: str) -> tuple:
    """Download SBML for a BioModels id; returns (sbml_text or None, source).

    source is "biomodels", "mirror", "not_found" (BioModels or the mirror answered and
    has no SBML for that id), "unreachable" (nothing answered) or "invalid_id".

    Two things this must not do, both previously observed against the live API:
    the '/{id}/download' form answers 200 with an HTML landing page, and omitting
    'filename' answers 200 with a ZIP archive. Neither is SBML, so each response
    is content-checked rather than trusted on status alone, and the form known to
    return SBML is tried first.
    """
    if not _VALID_MODEL_ID.match(model_id or ""):
        print(f"Rejected malformed BioModels id: {model_id!r}")
        return None, "invalid_id"

    attempts = [
        (f"{BIOMODELS_API_URL}/model/download/{model_id}", {"filename": f"{model_id}_url.xml"}),
        (f"{BIOMODELS_API_URL}/{model_id}/download", {"filename": f"{model_id}_url.xml"}),
    ]
    answered = False
    biomodels_answered = False
    for url, params in ([] if _biomodels_known_down() else attempts):
        try:
            response = requests.get(url, timeout=8, params=params)
        except Exception as e:
            print(f"Error fetching BioModel SBML from {url}: {type(e).__name__}")
            continue
        answered = biomodels_answered = True
        if response.status_code == 200 and _looks_like_sbml(response.text):
            return response.text, "biomodels"
    if not biomodels_answered and not _biomodels_known_down():
        _mark_biomodels_down()

    if _CURATED_BIOMD.match(model_id):
        for template in BIOMODELS_MIRROR_URLS:
            try:
                response = requests.get(template.format(mid=model_id), timeout=20)
            except Exception as e:
                print(f"Error fetching BioModel SBML from the mirror: {type(e).__name__}")
                continue
            answered = True
            if response.status_code == 200 and _looks_like_sbml(response.text):
                return response.text, "mirror"
    return None, ("not_found" if answered else "unreachable")


def fetch_biomodel_sbml(model_id: str) -> Optional[str]:
    """Download SBML content for a given BioModels ID (see fetch_biomodel_sbml_with_source)."""
    return fetch_biomodel_sbml_with_source(model_id)[0]


# Offline fallbacks, used only when a service does not answer. Every record carries
# "offline": True so the interface can say it is an offline example, and none carries a
# literature reference: the earlier set cited PMIDs that belong to unrelated papers
# (PMID 12345678 is the "Denpasar Declaration on Population and Development").
def get_mock_pathway_search(query: str) -> List[Dict[str, Any]]:
    query_lower = query.lower()
    if "egfr" in query_lower or "mapk" in query_lower or "signaling" in query_lower:
        # Names checked against Reactome's ContentService on 2026-09-28.
        return [
            {"id": "R-HSA-177929", "name": "Signaling by EGFR", "species": "Homo sapiens",
             "details": "Pathway", "offline": True},
            {"id": "R-HSA-5684996", "name": "MAPK1/MAPK3 signaling", "species": "Homo sapiens",
             "details": "Pathway", "offline": True},
            {"id": "R-HSA-5683057", "name": "MAPK family signaling cascades", "species": "Homo sapiens",
             "details": "Pathway", "offline": True},
        ]
    return []

def get_mock_reactions(pathway_id: str) -> List[Dict[str, Any]]:
    if pathway_id in ("R-HSA-177929", "R-HSA-5684996", "R-HSA-5683057"):
        # A simplified offline summary of the canonical cascade, not Reactome's own events.
        return [
            {"id": "offline-1", "name": "EGF binding to EGFR and receptor dimerization", "type": "reaction", "offline": True},
            {"id": "offline-2", "name": "EGFR autophosphorylation", "type": "reaction", "offline": True},
            {"id": "offline-3", "name": "Activated EGFR triggers SOS to activate RAS", "type": "reaction", "offline": True},
            {"id": "offline-4", "name": "RAS-GTP activates RAF", "type": "reaction", "offline": True},
            {"id": "offline-5", "name": "RAF phosphorylates and activates MEK", "type": "reaction", "offline": True},
            {"id": "offline-6", "name": "MEK phosphorylates and activates ERK", "type": "reaction", "offline": True}
        ]
    return []

def get_mock_string_network(proteins: List[str]) -> List[Dict[str, Any]]:
    # Offline placeholder: chains the typed proteins; not STRING data.
    interactions = []
    normalized = [p.upper() for p in proteins]
    for i in range(len(normalized) - 1):
        interactions.append({
            "source": normalized[i],
            "target": normalized[i+1],
            "score": None,
            "type": "association",
            "offline": True
        })
    return interactions


def get_mock_omnipath(proteins: List[str]) -> List[Dict[str, Any]]:
    """Offline example interactions (canonical EGFR/MAPK and TGF-beta steps), not OmniPath data."""
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
                "references": "",
                "sources_db": "offline example",
                "score": None,
                "offline": True
            })
    return interactions


def get_mock_signor(query: str) -> List[Dict[str, Any]]:
    """Offline example relations (canonical EGFR/MAPK cascade), not SIGNOR records."""
    q = query.lower()
    if "egfr" in q or "mapk" in q or "ras" in q:
        steps = [("EGF", "EGFR", "binding"), ("EGFR", "GRB2", "binding"), ("GRB2", "SOS1", "binding"),
                 ("SOS1", "HRAS", "guanine nucleotide exchange factor"), ("HRAS", "RAF1", "binding"),
                 ("RAF1", "MAP2K1", "phosphorylation"), ("MAP2K1", "MAPK1", "phosphorylation")]
        return [{"source": a, "target": b, "mechanism": m, "effect": "up-regulates activity",
                 "type": "activation", "pmid": "", "offline": True} for a, b, m in steps]
    return []


def get_mock_biomodels(query: str) -> List[Dict[str, Any]]:
    """Offline BioModels examples; ids and names checked against the curated SBML files."""
    q = query.lower()
    catalogue = [
        ("BIOMD0000000010", "Kholodenko2000 - Ultrasensitivity and negative feedback bring oscillations in MAPK cascade",
         ("mapk", "erk", "egfr", "signal", "oscillat", "feedback")),
        ("BIOMD0000000009", "Huang1996 - Ultrasensitivity in MAPK cascade", ("mapk", "erk", "ultrasens", "signal")),
        ("BIOMD0000000048", "Kholodenko1999 - EGFR signaling", ("egfr", "egf", "signal", "receptor")),
        ("BIOMD0000000019", "Schoeberl2002 - EGF MAPK", ("egfr", "egf", "mapk", "signal")),
        ("BIOMD0000000012", "Elowitz2000 - Repressilator", ("repressilator", "oscillat", "gene", "synthetic")),
        ("BIOMD0000000005", "Tyson1991 - Cell Cycle 6 var", ("cell cycle", "cycle", "mitosis", "cdc")),
        ("BIOMD0000000006", "Tyson1991 - Cell Cycle 2 var", ("cell cycle", "cycle", "mitosis", "cdc")),
        ("BIOMD0000000043", "Borghans1997 - Calcium Oscillation - Model 1", ("calcium", "ca2", "oscillat")),
        ("BIOMD0000000098", "Goldbeter1990_CalciumSpike_CICR", ("calcium", "ca2", "spike", "cicr")),
        ("BIOMD0000000035", "Vilar2002_Oscillator", ("circadian", "oscillat", "clock")),
        ("BIOMD0000000021", "Leloup1999_CircClock", ("circadian", "clock", "drosophila", "per")),
    ]
    hits = [(mid, name) for mid, name, keys in catalogue if any(k in q for k in keys)]
    if not hits:
        hits = [(mid, name) for mid, name, _ in catalogue[:3]]
    return [{"id": mid, "name": name,
             "description": "Offline example: BioModels did not respond to the search.",
             "format": "SBML", "submitter": "", "publication": "", "offline": True}
            for mid, name in hits]

