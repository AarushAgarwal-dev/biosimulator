"""Umulis et al. (2010) organism-scale BMP patterning model of the Drosophila embryo.

Umulis DM, Shimmi O, O'Connor MB, Othmer HG. "Organism-Scale Modeling of Early Drosophila
Patterning via Bone Morphogenetic Proteins". Developmental Cell 18:260-274 (2010),
doi:10.1016/j.devcel.2010.01.006.  Units throughout: nM, min, micrometres.

WHAT IS EXACT
-------------
* The reaction-transport equations: main-text Eqs 1-7 and Supplemental Eqs 7-21 (no feedback,
  receptor feedback) and 52-61 (the winning mechanism: positive feedback of a surface
  BMP-binding protein, SBP), written term for term in ``_reaction``.
* Every rate constant, diffusivity, production rate and fitted feedback parameter printed in
  Tables S1, S2, S3 and S8 (diffusivities converted from um^2/s to um^2/min).
* The scale-invariance conservation conditions, Supplemental Eqs 95-104.

WHAT THE PAPER DID NOT PUBLISH, AND HOW IT IS RESOLVED (each is listed in ASSUMPTIONS)
-------------------------------------------------------------------------------------
* The Sog source.  Supplemental Eq. 82 makes it a Fourier-sine fit of FISH images whose
  coefficients are not given, and Table S1 prints its magnitude as "1.36 uM*min^1".  Taken
  literally (1360 nM/min) with the 2010 Sog equation - which has NO Sog decay - Sog/Tsg floods
  the perivitelline space and no dorsal BMP stripe forms at all (check D1).  The default set
  therefore uses the Sog secretion rate the same group published for the direct predecessor
  model (Umulis et al. 2006 PNAS, Table 1: 400 nM/min), in the lateral neuroectoderm.
  The literal value is kept as the ``umulis2010_as_printed`` set so the finding is reproducible.
* Source domains.  Read from the paper's own statements and Fig. 4A ("prepattern distributions
  as they appear in the 3D model"): BMP in the dorsal-most 40 % of the circumference (p.260)
  plus the termini; Tsg in a central AP block spanning the DV height; Sog in the lateral
  neuroectoderm.  Case 1 of the paper (uniform Tkv and Tld) is used.
* Initial conditions ("determined from image analysis", p.266, not published): zero for every
  species except free receptor (= total receptor), as in the 2006 predecessor.

HOW THE SOLVER IS VERIFIED INDEPENDENTLY
----------------------------------------
The 2006 predecessor (Umulis, Serpe, O'Connor & Othmer, PNAS 103:11613) is FULLY specified in
its Supporting Information (Table 1, Eqs 4-8 and 28-32, domain sizes, 55 nodes).  The same
kernel and solver reproduce its published Fig. 13 curves (check S1), so any remaining
difference from the 2010 figures comes from the unpublished inputs above, not the numerics.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.sparse import linalg as splinalg

CITATION = (
    "Umulis DM, Shimmi O, O'Connor MB, Othmer HG (2010), Developmental Cell "
    "18:260-274, doi:10.1016/j.devcel.2010.01.006"
)
CITATION_2006 = (
    "Umulis DM, Serpe M, O'Connor MB, Othmer HG (2006), PNAS 103:11613-11618, "
    "doi:10.1073/pnas.0510398103 (Supporting Information)"
)
MECHANISMS = ("sbp", "none", "receptor")
PARAMETER_SETS = ("umulis2010", "umulis2010_as_printed", "umulis2006")
DEFAULT_PARAMETER_SET = "umulis2010"
SPECIES = ("B", "S", "T", "I", "IB", "C", "BC", "BCR", "BR")
DIFFUSING = ("B", "S", "T", "I", "IB")
SPECIES_NAMES = {
    "B": "Dpp/Scw heterodimer (free BMP)", "S": "Sog", "T": "Tsg", "I": "Sog/Tsg",
    "IB": "Sog/Tsg/BMP", "C": "surface BMP-binding protein (SBP)", "BC": "SBP/BMP",
    "BCR": "SBP/BMP/receptor", "BR": "BMP-bound receptor (signal, compared to pMad)",
}
PER_S = 60.0     # um^2/s -> um^2/min


def _entry(value: float, units: str, source: str, interpretation: str = "") -> dict[str, Any]:
    d: dict[str, Any] = {"value": float(value), "units": units, "source": source}
    if interpretation:
        d["interpretation"] = interpretation
    return d


_S1 = "Umulis 2010 Table S1 (Supplemental p.7)"
_S2 = "Umulis 2010 Table S2 (Supplemental p.8)"
_S3 = "Umulis 2010 Table S3 (Supplemental p.8)"
_S8 = "Umulis 2010 Table S8 (Supplemental p.12)"
_T06 = "Umulis 2006 PNAS Supporting Information, Table 1 (p.28)"

PHI_S_NOTE = (
    "Table S1 prints 'phi_S 1.36 uM*min^1'. Used literally (1360 nM/min) with the 2010 Sog "
    "equation (no Sog decay term) it floods the PV space with Sog/Tsg and abolishes the dorsal "
    "BMP stripe entirely (validation D1). The 2010 Sog source was a FISH-derived field "
    "(Supplemental Eq. 82) whose coefficients were not published, so the Sog secretion rate the "
    "same authors published for the predecessor model is used."
)

TABLE_S1 = {
    "k2": _entry(0.6, "nM^-1 min^-1", _S1, "Sog + Tsg binding"),
    "km2": _entry(3.6, "min^-1", _S1, "Sog/Tsg dissociation"),
    "k3": _entry(11.4, "nM^-1 min^-1", _S1, "Sog/Tsg + BMP binding"),
    "km3": _entry(0.36, "min^-1", _S1, "Sog/Tsg/BMP dissociation"),
    "k5": _entry(0.048, "nM^-1 min^-1", _S1, "BMP + receptor binding"),
    "km5": _entry(8.0, "min^-1", _S1, "BMP-receptor dissociation"),
    "D_S": _entry(45.0 * PER_S, "um^2 min^-1", _S1, "printed 45 um^2 s^-1, multiplied by 60"),
    "D_T": _entry(60.0 * PER_S, "um^2 min^-1", _S1, "printed 60 um^2 s^-1, multiplied by 60"),
    "D_B": _entry(65.7 * PER_S, "um^2 min^-1", _S1, "printed 65.7 um^2 s^-1, multiplied by 60"),
    "D_IB": _entry(38.0 * PER_S, "um^2 min^-1", _S1, "printed 38 um^2 s^-1, multiplied by 60"),
    "D_I": _entry(40.5 * PER_S, "um^2 min^-1", _S1, "printed 40.5 um^2 s^-1, multiplied by 60"),
    "phi_S": _entry(400.0, "nM min^-1", "Umulis 2006 PNAS Table 1 (Sog secretion) - see interpretation", PHI_S_NOTE),
    "phi_T": _entry(48.0, "nM min^-1", _S1, "Tsg secretion"),
    "phi_B": _entry(1.0, "nM min^-1", _S1, "BMP (Dpp/Scw) secretion"),
    "Tld": _entry(10.0, "nM", _S1, "provenance only; the fitted combined lambda*Tld is used"),
    "delta_T": _entry(0.05, "min^-1", _S1, "Tsg degradation"),
    "delta_E": _entry(0.03, "min^-1", _S1, "internalisation of surface complexes"),
    "delta_S": _entry(0.0, "min^-1", "Umulis 2010 Eq. 2 / Supp. Eq. 53 contain no Sog decay", "kept at zero for the 2010 sets"),
    "delta_B": _entry(0.0, "min^-1", "Umulis 2010 Eq. 1 / Supp. Eq. 52 contain no free-BMP decay", "kept at zero for the 2010 sets"),
}
PHI_S_AS_PRINTED = _entry(1360.0, "nM min^-1", _S1, "printed '1.36 uM*min^1', read as 1.36 uM/min")

MECH_2010 = {
    "none": {"Rtot": _entry(2386.0, "nM", _S2, "printed 2.386 uM"),
             "lambda_Tld": _entry(10.7, "min^-1", _S2, "Tld processing rate lambda*Tld")},
    "receptor": {"R_basal": _entry(2576.3, "nM", _S3, "printed 2.5763 uM"),
                 "lambda_Tld": _entry(11.0, "min^-1", _S3),
                 "Lambda": _entry(287.8, "nM", _S3, "receptor feedback amplitude"),
                 "K_h": _entry(18.1, "nM", _S3), "nu": _entry(2.0, "dimensionless", _S3, "fixed")},
    "sbp": {"Rtot": _entry(394.3, "nM", _S8), "lambda_Tld": _entry(31.35, "min^-1", _S8),
            "Lambda": _entry(24.45, "nM min^-1", _S8, "SBP production amplitude"),
            "K_h": _entry(61.83, "nM", _S8), "nu": _entry(2.0, "dimensionless", _S8, "fixed"),
            "k4": _entry(2.0, "nM^-1 min^-1", _S8, "BMP + SBP (not optimised; Cv-2 literature)"),
            "km4": _entry(4.0, "min^-1", _S8), "k6": _entry(1.0, "nM^-1 min^-1", _S8, "SBP/BMP + receptor"),
            "km6": _entry(20.0, "min^-1", _S8), "k7": _entry(0.25, "nM^-1 min^-1", _S8, "BMP/receptor + SBP"),
            "km7": _entry(20.0, "min^-1", _S8)},
}

PREPATTERN_2010 = {
    "dorsal_edge": _entry(0.40, "fraction of the DM->VM arc", "Umulis 2010 main text p.260",
                          "BMPs 'are secreted from a broad region making up the dorsal-most 40% of the embryo circumference'"),
    "dpp_terminal": _entry(0.10, "x/L at each pole", "Umulis 2010 Fig. 4A (Dpp prepattern)",
                           "dpp wraps the full circumference at both termini (surface model only)"),
    "sog_start": _entry(0.40, "fraction of the DM->VM arc", "Fig. 4A (Sog prepattern); 2006 predecessor NE domain",
                        "sog in the lateral neuroectoderm abutting the dorsal domain"),
    "sog_end": _entry(0.80, "fraction of the DM->VM arc", "Fig. 4A; 2006 predecessor (ventral 225-275 of 275 um has no Sog)",
                      "ventral ~20% (mesoderm) has no sog"),
    "sog_ap_start": _entry(0.10, "x/L", "Fig. 4A (Sog band stops short of the poles)", "surface model only"),
    "sog_ap_end": _entry(0.90, "x/L", "Fig. 4A", "surface model only"),
    "tsg_all_dv": _entry(1.0, "flag", "Fig. 4A (Tsg prepattern, lateral view)",
                         "Tsg source spans the full DV height inside a central AP block"),
    "tsg_ap_start": _entry(0.37, "x/L", "Fig. 4A (Tsg prepattern)", "surface model only; x/L=0.5 is inside"),
    "tsg_ap_end": _entry(0.63, "x/L", "Fig. 4A (Tsg prepattern)", "surface model only"),
    "smooth_width_um": _entry(5.0, "um", "numerical regularisation (one ~5 um nucleus), not from the paper",
                              "tanh smoothing of source boundaries so results are grid-independent"),
}

TABLE_2006 = {
    "k2": _entry(0.3, "nM^-1 min^-1", _T06), "km2": _entry(1.8, "min^-1", _T06),
    "k3": _entry(5.7, "nM^-1 min^-1", _T06), "km3": _entry(0.18, "min^-1", _T06),
    "k4": _entry(1.0, "nM^-1 min^-1", _T06), "km4": _entry(2.0, "min^-1", _T06),
    "k5": _entry(0.024, "nM^-1 min^-1", _T06), "km5": _entry(4.0, "min^-1", _T06),
    "k6": _entry(0.5, "nM^-1 min^-1", _T06), "km6": _entry(10.0, "min^-1", _T06),
    "k7": _entry(0.13, "nM^-1 min^-1", _T06), "km7": _entry(10.0, "min^-1", _T06),
    "D_S": _entry(50.0 * PER_S, "um^2 min^-1", _T06, "printed 50 um^2 s^-1"),
    "D_T": _entry(66.0 * PER_S, "um^2 min^-1", _T06, "printed 66 um^2 s^-1"),
    "D_B": _entry(73.0 * PER_S, "um^2 min^-1", _T06, "printed 73 um^2 s^-1"),
    "D_IB": _entry(42.0 * PER_S, "um^2 min^-1", _T06, "printed 42 um^2 s^-1"),
    "D_I": _entry(45.0 * PER_S, "um^2 min^-1", _T06, "printed 45 um^2 s^-1"),
    "phi_S": _entry(400.0, "nM min^-1", _T06), "phi_T": _entry(36.0, "nM min^-1", _T06),
    "phi_B": _entry(1e-3 * 1000.0 * 10.0 / 11.0, "nM min^-1", "2006 Supp. p.5 heterodimer module",
                    "(Vin/Vpv) * phi_D*phi_W/(phi_D+phi_W) with phi_D=1, phi_W=10 uM/min, Vin/Vpv=1e-3"),
    "Tld": _entry(6.0, "nM", _T06),
    "lambda_Tld": _entry(5.0 * 6.0, "min^-1", _T06, "lambda = 5 nM^-1 min^-1 times Tld = 6 nM"),
    "delta_S": _entry(0.15, "min^-1", _T06, "Sog degradation (present in the 2006 Eq. 5)"),
    "delta_T": _entry(0.05, "min^-1", _T06),
    "delta_B": _entry(0.0, "min^-1", "2006 Supp. section 4.7 (Case 4)",
                      "delta_B,eff = (delta_D/delta_tot)*delta_B with delta_D = 0 for endocytosis-only Case 4"),
    "delta_E": _entry(0.03, "min^-1", _T06),
    "Rtot": _entry(320.0, "nM", "2006 Supp. Fig. 13 caption (Case 4)"),
    "Lambda": _entry(24.0, "nM min^-1", _T06), "K_h": _entry(31.63, "nM", _T06),
    "nu": _entry(2.0, "dimensionless", _T06),
}
GEOMETRY_2006 = {
    "half_circumference_um": _entry(275.0, "um", "2006 Supp. p.8 ('the circumference is 550 um') and p.27 (half circumference, symmetry at DM/VM)"),
    "n_nodes": _entry(55, "nodes", "2006 Supp. p.27 ('55 node points in the half-width')"),
    "dorsal_edge_um": _entry(115.0, "um from DM", "2006 Supp. Figs 8, 12b-c, 13 (BMP/Tsg domain edge read off the plots)"),
    "sog_end_um": _entry(225.0, "um from DM", "2006 Supp. Fig. 12b-c domain bar (NE ends, ventral region has no source)"),
}

# Digitised published curves used by validation and shown in the UI.
PAPER_DATA = {
    "umulis2010_fig4F": {
        "source": "Umulis 2010 Fig. 4F (BR at x/Lx = 0.5), digitised; x-axis NE->DM carries no scale",
        "times_min": [15.0, 30.0, 45.0, 60.0], "dm_BR_nM": [16.8, 30.1, 35.3, 37.9], "ne_floor_BR_nM": 2.5,
    },
    "umulis2006_fig13": {
        "source": "Umulis 2006 PNAS Supp. Fig. 13 (Case 4, Rtot = 320 nM), digitised at the dorsal midline",
        "times_min": [15.0, 30.0, 45.0, 60.0, "s.s."],
        "dm_BMP_nM": [1.955, 2.867, 1.478, 0.964, 0.592],
        "dm_BR_nM": [3.95, 19.17, 33.24, 33.24, 31.12],
        "note": "At the midline the 45- and 60-min BR curves overlap (both ~33.2 nM).",
        "ss_stripe_half_width_um": 25.0,
    },
}

ASSUMPTIONS = [
    "Equations and Tables S1/S2/S3/S8 are used exactly; diffusivities converted from um^2/s to um^2/min.",
    PHI_S_NOTE,
    "Source domains follow Umulis 2010 p.260 and Fig. 4A: BMP in the dorsal-most 40% of the circumference "
    "(plus the termini on the surface), Tsg in a central AP block spanning the DV height, Sog in the "
    "lateral neuroectoderm (40%-80% of the DM->VM arc). Boundaries are tanh-smoothed over ~5 um.",
    "Case 1 of the paper: uniform Tkv (receptor) and uniform Tld (p.266).",
    "Initial conditions: all species zero except free receptor = total receptor (the 2010 image-derived "
    "initial conditions were not published; the 2006 predecessor used the same zero start).",
    "The 1D cross-section is the AP midline (x/L = 0.5) with the circular approximation "
    "L_half = pi*diameter/2 and mirror symmetry at the dorsal and ventral midlines.",
    "Surface species (C, BC, BCR, BR) do not diffuse; B, S, T, I and IB diffuse on the PV sheet.",
    "The 60-min horizon follows the 2010 Fig. 4E-F time axis (15-60 min).",
    "Observation, not a tuned choice: with uniform Tld (Case 1) and no Sog in the ventral mesoderm, "
    "shuttling also concentrates some BMP at the VENTRAL midline (BR about 18% of the dorsal value at "
    "60 min). The paper's training images were dorsal projections, so this region was not constrained by "
    "its data; extending Sog to the ventral midline removes it (and raises the dorsal value to ~34 nM).",
]

PERTURBATIONS: dict[str, dict[str, Any]] = {
    "wt": {"multipliers": {}, "description": "Wild type.", "citation": "Umulis 2010 Tables S1/S8"},
    "sog_het": {"multipliers": {"phi_S": 0.5}, "description": "sog+/-: Sog secretion halved.", "citation": "Umulis 2010 p.263-265, Fig. 3E-F"},
    "tsg_het": {"multipliers": {"phi_T": 0.5}, "description": "tsg+/-: Tsg secretion halved.", "citation": "Umulis 2010 Fig. 3B"},
    "scw_het": {"multipliers": {"phi_B": 0.9167}, "description": "scw+/-: heterodimer output (5/6)/(10/11).", "citation": "Umulis 2006 Supp. p.5; Shimmi et al. 2005"},
    "dpp_het": {"multipliers": {"phi_B": 0.5238}, "description": "dpp+/-: heterodimer output (5/10.5)/(10/11).", "citation": "Umulis 2006 Supp. p.5; Shimmi et al. 2005"},
    "sog_null": {"multipliers": {"phi_S": 0.0}, "description": "sog-/-: no Sog.", "citation": "O'Connor et al. 2006 Development review"},
    "tsg_null": {"multipliers": {"phi_T": 0.0}, "description": "tsg-/-: no Tsg.", "citation": "Umulis 2010 mutant training set"},
    "tld_null": {"multipliers": {"lambda_Tld": 0.0}, "description": "tld-/-: no Tld processing.", "citation": "Umulis 2006 Fig. 3a"},
    "invitro_k3": {"set": {"k3": 0.0168}, "description": "Biacore Chordin/BMP-2 on-rate 0.28e-3 nM^-1 s^-1.", "citation": "Umulis 2010 p.267-268, Fig. 5B"},
}

_EQUATIONS = {
    "B": r"\partial_t B=D_B\nabla^2B+\phi_B(\mathbf{x})-k_3 I B+k_{-3}IB+\lambda Tld\,IB-k_4 B C+k_{-4}BC-k_5 B R+k_{-5}BR",
    "S": r"\partial_t S=D_S\nabla^2S+\phi_S(\mathbf{x})-k_2 S T+k_{-2}I",
    "T": r"\partial_t T=D_T\nabla^2T+\phi_T(\mathbf{x})-k_2 S T+k_{-2}I+\lambda Tld\,IB-\delta_T T",
    "I": r"\partial_t I=D_I\nabla^2I+k_2 S T-k_{-2}I-k_3 I B+k_{-3}IB",
    "IB": r"\partial_t IB=D_{IB}\nabla^2IB+k_3 I B-k_{-3}IB-\lambda Tld\,IB",
    "C": r"\partial_t C=\frac{\Lambda BR^\nu}{K_h^\nu+BR^\nu}-k_4 B C+k_{-4}BC-k_7 BR\,C+k_{-7}BCR-\delta_E C",
    "BC": r"\partial_t BC=k_4 B C-k_{-4}BC-k_6 BC\,R+k_{-6}BCR-\delta_E BC",
    "BCR": r"\partial_t BCR=k_6 BC\,R+k_7 BR\,C-k_{-6}BCR-k_{-7}BCR-\delta_E BCR",
    "BR": r"\partial_t BR=k_5 B R-k_{-5}BR+k_{-7}BCR-k_7 BR\,C-\delta_E BR",
    "R": r"R_{tot}=R+BR+BCR",
}


def _flat_table() -> dict[str, dict[str, Any]]:
    table = dict(TABLE_S1)
    table["phi_S_as_printed"] = PHI_S_AS_PRINTED
    for mech, block in MECH_2010.items():
        for key, value in block.items():
            table[f"{mech}.{key}"] = value
    for key, value in PREPATTERN_2010.items():
        table[f"prepattern.{key}"] = value
    for key, value in TABLE_2006.items():
        table[f"umulis2006.{key}"] = value
    for key, value in GEOMETRY_2006.items():
        table[f"umulis2006.{key}"] = value
    return table


PARAMETER_TABLE: dict[str, dict[str, Any]] = _flat_table()


def model_info() -> dict[str, Any]:
    return {
        "citation": CITATION, "citation_2006": CITATION_2006,
        "mechanisms": list(MECHANISMS), "parameter_sets": list(PARAMETER_SETS),
        "default_parameter_set": DEFAULT_PARAMETER_SET,
        "perturbations": PERTURBATIONS, "species": list(SPECIES), "species_names": SPECIES_NAMES,
        "equations_latex": _EQUATIONS, "parameter_table": PARAMETER_TABLE,
        "assumptions": list(ASSUMPTIONS), "paper_data": PAPER_DATA,
        "geometry": {"length_ap_um": 400.0, "diameter_um": 180.0,
                     "surface": "prolate spheroid PV sheet (Supp. p.6-7)"},
    }


def _values(block: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    return {k: float(v["value"]) for k, v in block.items()}


def _base_params(mechanism: str, perturbation: str, overrides: Mapping[str, float] | None,
                 length_ap: float, conserve: bool, param_set: str = DEFAULT_PARAMETER_SET) -> dict[str, float]:
    if param_set not in PARAMETER_SETS:
        raise ValueError(f"unknown parameter_set {param_set!r}; valid names: {', '.join(PARAMETER_SETS)}")
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}; valid names: {', '.join(MECHANISMS)}")
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {perturbation!r}; valid names: {', '.join(PERTURBATIONS)}")
    if not (0.0 < float(length_ap) <= 2000.0):
        raise ValueError("length_ap must be in (0, 2000] um")
    zero_sbp = dict(k4=0.0, km4=0.0, k6=0.0, km6=0.0, k7=0.0, km7=0.0)
    if param_set == "umulis2006":
        if mechanism != "sbp":
            raise ValueError("the umulis2006 verification set implements the fully specified Case 4 ('sbp') only")
        if conserve or float(length_ap) != 400.0:
            raise ValueError("the umulis2006 set is a single fixed cross-section (no scaling)")
        p = _values(TABLE_2006)
        p.update(dorsal_edge=115.0 / 275.0, sog_start=115.0 / 275.0, sog_end=225.0 / 275.0,
                 dpp_terminal=0.0, sog_ap_start=0.0, sog_ap_end=1.0, tsg_all_dv=0.0,
                 tsg_ap_start=0.0, tsg_ap_end=1.0, smooth_width_um=1.0)
    else:
        p = _values(TABLE_S1)
        if param_set == "umulis2010_as_printed":
            p["phi_S"] = float(PHI_S_AS_PRINTED["value"])
        p.update(_values(MECH_2010[mechanism]))
        if mechanism == "none":
            p.update(Lambda=0.0, K_h=1.0, nu=2.0, **zero_sbp)
        elif mechanism == "receptor":
            p.update(**zero_sbp)
        p.update(_values(PREPATTERN_2010))
    pert = PERTURBATIONS[perturbation]
    for key, factor in pert.get("multipliers", {}).items():
        p[key] *= float(factor)
    for key, value in pert.get("set", {}).items():
        p[key] = float(value)
    if overrides:
        unknown = set(overrides) - set(p)
        if unknown:
            raise ValueError(f"unknown parameter override(s): {', '.join(sorted(unknown))}")
        for key, value in overrides.items():
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"override {key} must be finite and nonnegative")
            p[key] = value
    if conserve:
        # Supplemental Eqs 95-104: every source, Tld processing, Tsg loss, the feedback amplitude
        # and Hill scale, and the receptor concentration carry (L0/L)^2 with L0 = 400 um.
        q = (400.0 / float(length_ap)) ** 2
        for key in ("phi_B", "phi_S", "phi_T", "lambda_Tld", "delta_T", "Lambda", "K_h"):
            p[key] *= q
        p["R_basal" if mechanism == "receptor" else "Rtot"] *= q
    p["length_ap"] = float(length_ap)
    p["conserve"] = float(bool(conserve))
    return p


def _step_down(edge: float, x: np.ndarray, w: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh((edge - x) / w))


def _step_up(edge: float, x: np.ndarray, w: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh((x - edge) / w))


def _prepatterns(dv: np.ndarray, p: Mapping[str, float], dv_width: np.ndarray | float,
                 ap: np.ndarray | None = None, ap_width: float = 0.01):
    """Source fields (phi_B, phi_S, phi_T) from DV arc fraction and (surface only) AP fraction x/L."""
    dv = np.asarray(dv, dtype=float)
    dorsal = _step_down(p["dorsal_edge"], dv, dv_width)
    sog = _step_up(p["sog_start"], dv, dv_width) * _step_down(p["sog_end"], dv, dv_width)
    if p["tsg_all_dv"] > 0.5:
        tsg = np.ones_like(dv)
    else:
        tsg = dorsal.copy()
    bmp = dorsal
    if ap is not None:
        ap = np.asarray(ap, dtype=float)
        if p["dpp_terminal"] > 0.0:
            caps = np.clip(_step_down(p["dpp_terminal"], ap, ap_width)
                           + _step_up(1.0 - p["dpp_terminal"], ap, ap_width), 0.0, 1.0)
            bmp = 1.0 - (1.0 - dorsal) * (1.0 - caps)
        sog = sog * _step_up(p["sog_ap_start"], ap, ap_width) * _step_down(p["sog_ap_end"], ap, ap_width)
        if p["tsg_all_dv"] > 0.5:
            tsg = tsg * _step_up(p["tsg_ap_start"], ap, ap_width) * _step_down(p["tsg_ap_end"], ap, ap_width)
    return float(p["phi_B"]) * bmp, float(p["phi_S"]) * sog, float(p["phi_T"]) * tsg


def _reaction(y: np.ndarray, p: Mapping[str, float], mechanism: str,
              production: tuple[np.ndarray, np.ndarray, np.ndarray], jacobian: bool = False):
    """Pointwise reaction kernel shared by every geometry; optional exact local Jacobian."""
    B, S, T, I, IB, C, BC, BCR, BR = y
    m = B.size
    phi_B, phi_S, phi_T = production
    k2, km2, k3, km3 = p["k2"], p["km2"], p["k3"], p["km3"]
    k4, km4, k5, km5 = p["k4"], p["km4"], p["k5"], p["km5"]
    k6, km6, k7, km7 = p["k6"], p["km6"], p["k7"], p["km7"]
    lam, dT, dE = p["lambda_Tld"], p["delta_T"], p["delta_E"]
    dS, dB = p.get("delta_S", 0.0), p.get("delta_B", 0.0)
    if mechanism == "receptor":
        nu, kh = p["nu"], p["K_h"]
        brp = np.maximum(BR, 0.0)
        den = kh ** nu + brp ** nu
        R = p["R_basal"] + p["Lambda"] * brp ** nu / den - BR
        dR_dBR = p["Lambda"] * nu * kh ** nu * brp ** (nu - 1.0) / den ** 2 - 1.0
        dR_dBCR = np.zeros(m)
    else:
        R = p["Rtot"] - BR - (BCR if mechanism == "sbp" else 0.0)
        dR_dBR = -np.ones(m)
        dR_dBCR = -np.ones(m) if mechanism == "sbp" else np.zeros(m)
    r = np.zeros_like(y)
    st = k2 * S * T
    ibind = k3 * I * B
    r[0] = phi_B - ibind + km3 * IB + lam * IB - k4 * B * C + km4 * BC - k5 * B * R + km5 * BR - dB * B
    r[1] = phi_S - st + km2 * I - dS * S
    r[2] = phi_T - st + km2 * I + lam * IB - dT * T
    r[3] = st - km2 * I - ibind + km3 * IB
    r[4] = ibind - km3 * IB - lam * IB
    if mechanism == "sbp":
        nu, kh = p["nu"], p["K_h"]
        brp = np.maximum(BR, 0.0)
        den = kh ** nu + brp ** nu
        r[5] = p["Lambda"] * brp ** nu / den - k4 * B * C + km4 * BC - k7 * BR * C + km7 * BCR - dE * C
        r[6] = k4 * B * C - km4 * BC - k6 * BC * R + km6 * BCR - dE * BC
        r[7] = k6 * BC * R + k7 * BR * C - km6 * BCR - km7 * BCR - dE * BCR
        r[8] = k5 * B * R - km5 * BR + km7 * BCR - k7 * BR * C - dE * BR
    else:
        r[8] = k5 * B * R - km5 * BR - dE * BR
    if not jacobian:
        return r
    J = np.zeros((len(SPECIES), len(SPECIES), m), dtype=float)
    J[0, 0] = -k3 * I - k4 * C - k5 * R - dB
    J[0, 3] = -k3 * B
    J[0, 4] = km3 + lam
    J[0, 5] = -k4 * B
    J[0, 6] = km4
    J[0, 7] = -k5 * B * dR_dBCR
    J[0, 8] = -k5 * B * dR_dBR + km5
    J[1, 1] = -k2 * T - dS; J[1, 2] = -k2 * S; J[1, 3] = km2
    J[2, 1] = -k2 * T; J[2, 2] = -k2 * S - dT; J[2, 3] = km2; J[2, 4] = lam
    J[3, 0] = -k3 * I; J[3, 1] = k2 * T; J[3, 2] = k2 * S; J[3, 3] = -km2 - k3 * B; J[3, 4] = km3
    J[4, 0] = k3 * I; J[4, 3] = k3 * B; J[4, 4] = -km3 - lam
    if mechanism == "sbp":
        dh = p["nu"] * p["K_h"] ** p["nu"] * brp ** (p["nu"] - 1.0) / den ** 2
        J[5, 0] = -k4 * C; J[5, 5] = -k4 * B - k7 * BR - dE; J[5, 6] = km4; J[5, 7] = km7
        J[5, 8] = p["Lambda"] * dh - k7 * C
        J[6, 0] = k4 * C; J[6, 5] = k4 * B; J[6, 6] = -km4 - k6 * R - dE
        J[6, 7] = -k6 * BC * dR_dBCR + km6; J[6, 8] = -k6 * BC * dR_dBR
        J[7, 5] = k7 * BR; J[7, 6] = k6 * R; J[7, 7] = k6 * BC * dR_dBCR - km6 - km7 - dE
        J[7, 8] = k6 * BC * dR_dBR + k7 * C
        J[8, 0] = k5 * R; J[8, 5] = -k7 * BR; J[8, 7] = k5 * B * dR_dBCR + km7
        J[8, 8] = k5 * B * dR_dBR - km5 - k7 * C - dE
    else:
        J[8, 0] = k5 * R
        J[8, 7] = k5 * B * dR_dBCR
        J[8, 8] = k5 * B * dR_dBR - km5 - dE
    return r, J


def _cross_section_laplacian(n: int, length: float) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Second-order Neumann (mirror-symmetric) Laplacian on [0, length] with n nodes."""
    dx = length / (n - 1)
    main = np.full(n, -2.0 / dx ** 2)
    off = np.full(n - 1, 1.0 / dx ** 2)
    L = sparse.diags((off, main, off), (-1, 0, 1), shape=(n, n), format="lil")
    L[0, 1] = 2.0 / dx ** 2
    L[-1, -2] = 2.0 / dx ** 2
    s = np.linspace(0.0, length, n)
    weights = np.full(n, dx)
    weights[[0, -1]] = 0.5 * dx
    return L.tocsr(), weights, s


def _surface_laplacian(a: float, b: float, nu: int, nv: int) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, np.ndarray]]:
    """Finite-volume Laplace-Beltrami on the half prolate spheroid (v in [0, pi], DM -> VM)."""
    du, dv = math.pi / nu, math.pi / nv
    u = (np.arange(nu) + 0.5) * du
    v = (np.arange(nv) + 0.5) * dv
    hu = np.sqrt(a * a * np.sin(u) ** 2 + b * b * np.cos(u) ** 2)
    hv = b * np.sin(u)
    area = (hu[:, None] * hv[:, None] * du * dv * np.ones((1, nv))).ravel()
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    diag = np.zeros(nu * nv)

    def edge(pidx: int, qidx: int, conductance: float) -> None:
        vp, vq = conductance / area[pidx], conductance / area[qidx]
        rows.extend((pidx, qidx)); cols.extend((qidx, pidx)); vals.extend((vp, vq))
        diag[pidx] -= vp
        diag[qidx] -= vq

    for i in range(nu - 1):
        uf = (i + 1) * du
        huf = math.sqrt(a * a * math.sin(uf) ** 2 + b * b * math.cos(uf) ** 2)
        g = (b * math.sin(uf) / huf) * (dv / du)
        for j in range(nv):
            edge(i * nv + j, (i + 1) * nv + j, g)
    for i in range(nu):
        g = (hu[i] / hv[i]) * (du / dv)
        for j in range(nv - 1):
            edge(i * nv + j, i * nv + j + 1, g)
    idx = np.arange(nu * nv)
    rows.extend(idx.tolist()); cols.extend(idx.tolist()); vals.extend(diag.tolist())
    L = sparse.coo_matrix((vals, (rows, cols)), shape=(nu * nv, nu * nv)).tocsr()
    U, V = np.meshgrid(u, v, indexing="ij")
    grid = {"u": u, "v": v, "x": (a * np.cos(U)).ravel(), "y": (b * np.sin(U) * np.cos(V)).ravel(),
            "z": (b * np.sin(U) * np.sin(V)).ravel(), "hu": hu, "hv": hv}
    return L, area, grid


def _save_times(t_end: float, save_times: Sequence[float] | None) -> np.ndarray:
    if not (0.0 < float(t_end) <= 600.0):
        raise ValueError("t_end must be in (0, 600] min")
    if save_times is None:
        vals = [x for x in (0.0, 15.0, 30.0, 45.0, 60.0) if x <= t_end]
    else:
        vals = [float(x) for x in save_times]
        if any(not math.isfinite(x) or x < 0.0 or x > t_end for x in vals):
            raise ValueError("save_times must be finite and within [0, t_end]")
    vals.extend((0.0, float(t_end)))
    return np.asarray(sorted(set(vals)), dtype=float)


def _assemble_reaction_jac(local: np.ndarray, m: int) -> sparse.csr_matrix:
    ns = len(SPECIES)
    rr, cc, vv = [], [], []
    idx = np.arange(m)
    for a in range(ns):
        for b in range(ns):
            values = local[a, b]
            nz = values != 0.0
            if np.any(nz):
                ii = idx[nz]
                rr.append(ii * ns + a); cc.append(ii * ns + b); vv.append(values[nz])
    if not rr:
        return sparse.csr_matrix((ns * m, ns * m))
    return sparse.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                             shape=(ns * m, ns * m)).tocsr()


def _solve(L: sparse.csr_matrix, weights: np.ndarray, production, p: Mapping[str, float],
           mechanism: str, times: np.ndarray, time_budget_s: float | None = None):
    """Method of lines + BDF with an analytic sparse Jacobian; returns (frames, diagnostics)."""
    m = L.shape[0]
    ns = len(SPECIES)
    nstate = ns * m
    y_phys0 = np.zeros(nstate)
    if mechanism == "receptor":
        pass                                    # R is algebraic from BR; nothing to seed
    diffusivities = sparse.diags([p["D_B"], p["D_S"], p["D_T"], p["D_I"], p["D_IB"], 0.0, 0.0, 0.0, 0.0], format="csr")
    diffusion = sparse.kron(L, diffusivities, format="csr")
    start = time.perf_counter()
    # A fill-reducing ordering chosen once from the initial Jacobian; BDF then uses standard LU.
    _, local0 = _reaction(y_phys0.reshape(m, ns).T, p, mechanism, production, jacobian=True)
    jac0 = diffusion + _assemble_reaction_jac(local0, m)
    ordering = splinalg.splu((sparse.eye(nstate, format="csc") - 1e-3 * jac0).tocsc(), permc_spec="MMD_ATA")
    perm = np.asarray(ordering.perm_c, dtype=int)
    inv_perm = np.argsort(perm)
    del ordering, jac0, local0
    source = np.column_stack([production[0], production[1], production[2]] + [np.zeros(m)] * 6)

    def rhs(_t: float, yy: np.ndarray) -> np.ndarray:
        phys = yy[inv_perm]
        fields = phys.reshape(m, ns).T
        out = diffusion @ phys + _reaction(fields, p, mechanism, production).T.ravel()
        return out[perm]

    def jac(_t: float, yy: np.ndarray) -> sparse.csr_matrix:
        fields = yy[inv_perm].reshape(m, ns).T
        _, local = _reaction(fields, p, mechanism, production, jacobian=True)
        return (diffusion + _assemble_reaction_jac(local, m))[perm, :][:, perm].tocsr()

    del source
    sol = solve_ivp(rhs, (0.0, float(times[-1])), y_phys0[perm], method="BDF", t_eval=times,
                    jac=jac, rtol=1e-6, atol=1e-9, dense_output=True)
    if not sol.success:
        raise RuntimeError(f"BDF integration failed: {sol.message}")
    arr = sol.y.T[:, inv_perm].reshape(len(times), m, ns).transpose(0, 2, 1)
    # Ligand balance: d/dt(total ligand) = int(phi_B) - delta_E*int(BC+BCR+BR) - delta_B*int(B).
    ligand_idx = [0, 4, 6, 7, 8]
    actual = np.asarray([float(np.dot(weights, frame[ligand_idx].sum(axis=0))) for frame in arr])
    dense_t = np.unique(np.concatenate((np.linspace(0.0, float(times[-1]), max(2, int(times[-1] * 10) + 1)), times)))
    rates = np.empty(dense_t.size)
    total_source = float(np.dot(weights, production[0]))
    for lo in range(0, dense_t.size, 64):
        hi = min(lo + 64, dense_t.size)
        dense = sol.sol(dense_t[lo:hi]).T[:, inv_perm].reshape(hi - lo, m, ns)
        bound = dense[:, :, 6] + dense[:, :, 7] + dense[:, :, 8]
        rates[lo:hi] = (total_source - p["delta_E"] * (bound @ weights)
                        - p.get("delta_B", 0.0) * (dense[:, :, 0] @ weights))
    cumulative = np.zeros_like(dense_t)
    cumulative[1:] = np.cumsum(0.5 * (rates[1:] + rates[:-1]) * np.diff(dense_t))
    expected = np.interp(times, dense_t, cumulative)
    mbe = np.abs(actual - expected) / np.maximum(np.abs(expected), 1.0)
    elapsed = time.perf_counter() - start
    if time_budget_s is not None and elapsed > time_budget_s:
        raise TimeoutError(f"solve took {elapsed:.2f}s, exceeding time_budget_s={time_budget_s}")
    max_abs = max(float(np.max(np.abs(arr))), 1e-30)
    diagnostics = {
        "success": True, "message": str(sol.message), "method": "BDF",
        "analytic_sparse_jacobian": True, "state_ordering": "node-major + MMD_ATA",
        "rtol": 1e-6, "atol": 1e-9, "nfev": int(sol.nfev), "njev": int(sol.njev), "nlu": int(sol.nlu),
        "wall_time_s": elapsed, "min_value": float(np.min(arr)), "max_abs_value": max_abs,
        "positivity_pass": bool(np.min(arr) > -1e-6 * max_abs),
        "ligand_mass_actual": actual.tolist(), "ligand_mass_expected": expected.tolist(),
        "ligand_mass_balance_relative_error": float(np.max(mbe)),
        "ligand_mass_balance_pass": bool(np.max(mbe) < 1e-3),
    }
    return arr, diagnostics


def _crossings_width(coord: np.ndarray, values: np.ndarray, threshold: float, mirror: bool = True) -> float:
    """Total length where values >= threshold (linear crossings); doubled for the mirror half."""
    if not np.any(values >= threshold):
        return 0.0
    width = 0.0
    for i in range(len(coord) - 1):
        x0, x1, y0, y1 = coord[i], coord[i + 1], values[i], values[i + 1]
        if y0 >= threshold and y1 >= threshold:
            width += x1 - x0
        elif (y0 >= threshold) != (y1 >= threshold):
            xc = x0 + (threshold - y0) / (y1 - y0) * (x1 - x0)
            width += (xc - x0) if y0 >= threshold else (x1 - xc)
    return float(width * (2.0 if mirror else 1.0))


def _profile_readout(coord: np.ndarray, br: np.ndarray, reference_max: float | None = None,
                     circumference: float | None = None) -> dict[str, Any]:
    peak = float(np.max(br))
    imax = int(np.argmax(br))
    ref = peak if reference_max is None else float(reference_max)
    widths_own, widths_ref = {}, {}
    for threshold in (0.2, 0.4, 0.5):
        wo = _crossings_width(coord, br, threshold * peak)
        wr = _crossings_width(coord, br, threshold * ref)
        widths_own[str(threshold)] = {"um": wo, "cells_5um": wo / 5.0}
        widths_ref[str(threshold)] = {"um": wr, "cells_5um": wr / 5.0}
    off_dm = imax > 0 and coord[imax] > max(5.0, coord[1] if len(coord) > 1 else 0.0)
    out = {
        "peak": peak, "peak_location_um_from_dm": float(coord[imax]), "dm_value": float(br[0]),
        "normalised_BR": (br / peak).tolist() if peak > 0 else np.zeros_like(br).tolist(),
        "widths_own_max": widths_own, "widths_reference_max": widths_ref,
        "dm_split": bool(off_dm and len(br) > 1 and br[0] < br[1]),
    }
    if circumference:
        out["fwhm_percent_circumference"] = 100.0 * widths_own["0.5"]["um"] / circumference
    return out


def simulate_cross_section(mechanism: str = "sbp", perturbation: str = "wt",
                           overrides: Mapping[str, float] | None = None, n_nodes: int | None = None,
                           t_end: float = 60.0, save_times: Sequence[float] | None = None,
                           length_ap: float = 400.0, conserve: bool = False,
                           parameter_set: str = DEFAULT_PARAMETER_SET) -> dict[str, Any]:
    """1D DV cross-section at the AP midline (DM at s = 0, VM at s = half-circumference)."""
    if n_nodes is None:
        n_nodes = int(GEOMETRY_2006["n_nodes"]["value"]) if parameter_set == "umulis2006" else 57
    if not isinstance(n_nodes, int) or not (9 <= n_nodes <= 801):
        raise ValueError("n_nodes must be an integer in [9, 801]")
    p = _base_params(mechanism, perturbation, overrides, length_ap, conserve, parameter_set)
    if parameter_set == "umulis2006":
        half = float(GEOMETRY_2006["half_circumference_um"]["value"])
    else:
        half = math.pi * (180.0 * length_ap / 400.0) / 2.0
    L, weights, s = _cross_section_laplacian(n_nodes, half)
    production = _prepatterns(s / half, p, max(p["smooth_width_um"] / half, 1e-6))
    times = _save_times(t_end, save_times)
    arr, diagnostics = _solve(L, weights, production, p, mechanism, times)
    circumference = 2.0 * half
    by_time = {str(float(t)): _profile_readout(s, arr[k, 8], circumference=circumference)
               for k, t in enumerate(times)}
    return {
        "grid": {"kind": "cross_section", "s_um": s.tolist(), "n_nodes": n_nodes,
                 "half_circumference_um": half, "circumference_um": circumference,
                 "quadrature_weights_um": weights.tolist()},
        "times": times.tolist(),
        "fields": {name: arr[:, i, :].tolist() for i, name in enumerate(SPECIES)},
        "sources": {"phi_B": production[0].tolist(), "phi_S": production[1].tolist(), "phi_T": production[2].tolist()},
        "readouts": {"by_time": by_time, "final": by_time[str(float(times[-1]))]},
        "diagnostics": diagnostics, "params": {k: float(v) for k, v in p.items()},
        "mechanism": mechanism, "perturbation": perturbation, "parameter_set": parameter_set,
        "assumptions": list(ASSUMPTIONS),
    }


def simulate_surface(mechanism: str = "sbp", perturbation: str = "wt",
                     overrides: Mapping[str, float] | None = None, nu: int = 64, nv: int = 40,
                     t_end: float = 60.0, save_times: Sequence[float] | None = None,
                     length_ap: float = 400.0, conserve: bool = False,
                     time_budget_s: float | None = None,
                     parameter_set: str = DEFAULT_PARAMETER_SET) -> dict[str, Any]:
    """Organism-scale solve on the PV sheet of a prolate spheroid (400 x 180 um by default)."""
    if not isinstance(nu, int) or not isinstance(nv, int) or not (8 <= nu <= 160 and 8 <= nv <= 120):
        raise ValueError("nu and nv must be integers with nu in [8,160], nv in [8,120]")
    if time_budget_s is not None and time_budget_s <= 0:
        raise ValueError("time_budget_s must be positive")
    if parameter_set == "umulis2006":
        raise ValueError("the umulis2006 verification set is a 1D cross-section model")
    p = _base_params(mechanism, perturbation, overrides, length_ap, conserve, parameter_set)
    a, b = length_ap / 2.0, 90.0 * length_ap / 400.0
    L_full, area, g = _surface_laplacian(a, b, nu, nv)
    dv = np.tile(g["v"] / math.pi, nu)
    ap = np.repeat((1.0 - np.cos(g["u"])) / 2.0, nv)            # x/L, 0 = anterior pole
    local_half = np.repeat(math.pi * b * np.sin(g["u"]), nv)
    dv_width = np.maximum(p["smooth_width_um"] / np.maximum(local_half, 1e-12), 1e-6)
    production_full = _prepatterns(dv, p, dv_width, ap=ap, ap_width=max(p["smooth_width_um"] / length_ap, 1e-6))
    times = _save_times(t_end, save_times)
    if nu % 2 == 0:
        # Geometry, prepatterns, parameters and initial state are mirror-symmetric about x/L = 0.5,
        # so solve the anterior half with a no-flux midplane and reflect.
        nh, mh = nu // 2, (nu // 2) * nv
        L = L_full[:mh, :mh].tolil()
        for j in range(nv):
            pidx, qidx = (nh - 1) * nv + j, nh * nv + j
            L[pidx, pidx] += L_full[pidx, qidx]
        production = tuple(src[:mh] for src in production_full)
        half_arr, diagnostics = _solve(L.tocsr(), 2.0 * area[:mh], production, p, mechanism, times, time_budget_s)
        grid4 = half_arr.reshape(len(times), len(SPECIES), nh, nv)
        arr = np.concatenate((grid4, grid4[:, :, ::-1, :]), axis=2).reshape(len(times), len(SPECIES), nu * nv)
        diagnostics["ap_symmetry_reduction"] = True
    else:
        arr, diagnostics = _solve(L_full, area, production_full, p, mechanism, times, time_budget_s)
        diagnostics["ap_symmetry_reduction"] = False
    diagnostics["surface_area_full_numeric_um2"] = float(2.0 * np.sum(area))
    br = arr[-1, 8].reshape(nu, nv)
    widths, split_map = [], []
    for i in range(nu):
        coord = b * math.sin(g["u"][i]) * g["v"]
        rr = _profile_readout(coord, br[i], circumference=2.0 * math.pi * b * math.sin(g["u"][i]))
        widths.append(rr["widths_own_max"]["0.5"]["um"])
        split_map.append(rr["dm_split"])
    dorsal = br[:, 0]
    dmax = float(np.max(dorsal))
    peak_i = int(np.argmax(dorsal))
    readouts = {
        "final": _profile_readout(b * math.sin(g["u"][nu // 2]) * g["v"], br[nu // 2],
                                  circumference=2.0 * math.pi * b * math.sin(g["u"][nu // 2])),
        "dorsal_midline": {"ap_x_um": (a * np.cos(g["u"])).tolist(), "ap_fraction": ((1 - np.cos(g["u"])) / 2).tolist(),
                           "BR": dorsal.tolist(),
                           "normalised_BR": (dorsal / dmax).tolist() if dmax > 0 else np.zeros_like(dorsal).tolist(),
                           "min_max_ratio": float(np.min(dorsal) / dmax) if dmax > 0 else 0.0,
                           "peak_x_um": float(a * math.cos(g["u"][peak_i])),
                           "peak_ap_fraction": float((1 - math.cos(g["u"][peak_i])) / 2)},
        "width_fwhm_um_vs_ap": widths, "split_map_vs_ap": split_map,
    }
    return {
        "grid": {"kind": "prolate_spheroid_half_surface", "nu": nu, "nv": nv, "a_um": a, "b_um": b,
                 "u": g["u"].tolist(), "v": g["v"].tolist(), "x_um": g["x"].reshape(nu, nv).tolist(),
                 "y_um": g["y"].reshape(nu, nv).tolist(), "z_um": g["z"].reshape(nu, nv).tolist(),
                 "cell_area_um2": area.reshape(nu, nv).tolist()},
        "times": times.tolist(),
        "fields": {name: arr[:, i, :].reshape(len(times), nu, nv).tolist() for i, name in enumerate(SPECIES)},
        "readouts": readouts, "diagnostics": diagnostics, "params": {k: float(v) for k, v in p.items()},
        "mechanism": mechanism, "perturbation": perturbation, "parameter_set": parameter_set,
        "assumptions": list(ASSUMPTIONS),
    }


def grid_convergence(n_coarse: int = 57, n_fine: int = 113, **kwargs: Any) -> dict[str, Any]:
    options = dict(kwargs)
    t_end = float(options.pop("t_end", 60.0))
    options.pop("save_times", None)
    coarse = simulate_cross_section(n_nodes=n_coarse, t_end=t_end, save_times=[t_end], **options)
    fine = simulate_cross_section(n_nodes=n_fine, t_end=t_end, save_times=[t_end], **options)
    sc, sf = np.asarray(coarse["grid"]["s_um"]), np.asarray(fine["grid"]["s_um"])
    bc, bf = np.asarray(coarse["fields"]["BR"][-1]), np.asarray(fine["fields"]["BR"][-1])
    fi = np.interp(sc, sf, bf)
    rel = float(np.linalg.norm(bc - fi) / max(np.linalg.norm(fi), 1e-30))
    return {"coarse_nodes": n_coarse, "fine_nodes": n_fine, "relative_l2_BR": rel, "pass": rel < 0.03}


def _row(key: str, claim: str, citation: str, how: str, measured: Mapping[str, Any], passed: bool | None,
         kind: str = "paper_claim") -> dict[str, Any]:
    return {"id": key, "kind": kind, "claim": claim, "citation": citation, "how_measured": how,
            "measured": dict(measured), "pass": None if passed is None else bool(passed)}


def _rel(model: float, paper: float) -> float:
    return (model - paper) / paper


def validate(include_surface: bool = False) -> dict[str, Any]:
    """Check the implementation against published curves and claims. Nothing here is tuned."""
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []

    # --- S1: solver verification against the fully specified 2006 predecessor ------------------
    ref = PAPER_DATA["umulis2006_fig13"]
    r06 = simulate_cross_section(parameter_set="umulis2006", t_end=600.0, save_times=[15.0, 30.0, 45.0, 60.0, 480.0, 600.0])
    t06 = r06["times"]
    dmB = {t: r06["fields"]["B"][t06.index(t)][0] for t in (15.0, 30.0, 45.0, 60.0, 600.0)}
    dmR = {t: r06["fields"]["BR"][t06.index(t)][0] for t in (15.0, 30.0, 45.0, 60.0, 600.0)}
    steady_drift = abs(r06["fields"]["BR"][t06.index(600.0)][0] - r06["fields"]["BR"][t06.index(480.0)][0]) / max(dmR[600.0], 1e-30)
    s06 = np.asarray(r06["grid"]["s_um"])
    br_ss = np.asarray(r06["fields"]["BR"][-1])
    half_width = _crossings_width(s06, br_ss, 0.5 * br_ss.max(), mirror=False)
    keys = [30.0, 45.0, 60.0, 600.0]
    errs_B = {str(t if t != 600.0 else "s.s."): _rel(dmB[t], ref["dm_BMP_nM"][i + 1]) for i, t in enumerate(keys)}
    errs_R = {str(t if t != 600.0 else "s.s."): _rel(dmR[t], ref["dm_BR_nM"][i + 1]) for i, t in enumerate(keys)}
    ok = all(abs(e) <= 0.12 for e in list(errs_B.values()) + list(errs_R.values())) and abs(half_width - 25.0) <= 10.0
    rows.append(_row(
        "S1", "The solver reproduces the fully specified 2006 predecessor model (Case 4)",
        "Umulis 2006 PNAS Supp. Fig. 13 + Table 1", "Same kernel and BDF solver with the 2006 Table 1 parameters; "
        "dorsal-midline BMP and BR vs the digitised figure (pass: every point from 30 min to steady state within 12%, "
        "steady-state stripe half-width within 10 um of 25 um)",
        {"model_dm_BMP_nM": {"15": dmB[15.0], "30": dmB[30.0], "45": dmB[45.0], "60": dmB[60.0], "s.s.": dmB[600.0]},
         "paper_dm_BMP_nM": dict(zip(["15", "30", "45", "60", "s.s."], ref["dm_BMP_nM"])),
         "model_dm_BR_nM": {"15": dmR[15.0], "30": dmR[30.0], "45": dmR[45.0], "60": dmR[60.0], "s.s.": dmR[600.0]},
         "paper_dm_BR_nM": dict(zip(["15", "30", "45", "60", "s.s."], ref["dm_BR_nM"])),
         "relative_error_BMP": errs_B, "relative_error_BR": errs_R,
         "steady_state_half_width_um": half_width, "paper_half_width_um": 25.0,
         "steady_state_drift_480_to_600_min": steady_drift,
         "known_mismatch_15_min": "at 15 min the paper's profile is already higher (BMP 1.96 vs model "
                                  f"{dmB[15.0]:.2f} nM); from 30 min on the curves agree",
         "mass_balance_relative_error": r06["diagnostics"]["ligand_mass_balance_relative_error"]},
        ok, kind="solver_verification"))

    # --- D1: the printed Sog secretion cannot be the value used --------------------------------
    printed = simulate_cross_section(parameter_set="umulis2010_as_printed", save_times=[60.0])
    rp = printed["readouts"]["final"]
    rows.append(_row(
        "D1", "Table S1's printed Sog secretion (1.36 uM/min), used literally, abolishes the dorsal stripe",
        "Umulis 2010 Table S1 vs main text p.260", "Same model with phi_S = 1360 nM/min: 60-min BR FWHM as % of circumference",
        {"fwhm_percent_circumference": rp["fwhm_percent_circumference"], "peak_BR_nM": rp["peak"],
         "dm_BR_nM": rp["dm_value"]},
        rp["fwhm_percent_circumference"] >= 90.0, kind="diagnostic"))

    # --- default 2010 set ------------------------------------------------------------------------
    wt = simulate_cross_section(save_times=[15.0, 30.0, 45.0, 60.0])
    s = np.asarray(wt["grid"]["s_um"])
    circ = wt["grid"]["circumference_um"]
    frames = {t: np.asarray(wt["fields"]["BR"][wt["times"].index(t)]) for t in (15.0, 30.0, 45.0, 60.0)}
    r30 = _profile_readout(s, frames[30.0], circumference=circ)
    r60 = _profile_readout(s, frames[60.0], circumference=circ)
    w30, w60 = r30["widths_own_max"]["0.5"]["um"], r60["widths_own_max"]["0.5"]["um"]

    f4f = PAPER_DATA["umulis2010_fig4F"]
    model_dm = [float(frames[t][0]) for t in (15.0, 30.0, 45.0, 60.0)]
    ne_60 = float(np.interp(0.6 * s[-1], s, frames[60.0]))
    rows.append(_row(
        "F1", "Dorsal-midline BR level at 60 min (Fig. 4F: 37.9 nM)", "Umulis 2010 Fig. 4F",
        "Model DM BR at 60 min vs the digitised figure (pass: within 25%)",
        {"model_nM": model_dm[3], "paper_nM": f4f["dm_BR_nM"][3], "relative_error": _rel(model_dm[3], f4f["dm_BR_nM"][3])},
        abs(_rel(model_dm[3], f4f["dm_BR_nM"][3])) <= 0.25, kind="paper_figure"))
    rel_all = [_rel(mv, pv) for mv, pv in zip(model_dm, f4f["dm_BR_nM"])]
    rows.append(_row(
        "F2", "Dorsal-midline BR time course (Fig. 4F: 16.8, 30.1, 35.3, 37.9 nM at 15-60 min)", "Umulis 2010 Fig. 4F",
        "Each time point within 25% (depends on the unpublished Sog field and initial conditions)",
        {"model_nM": dict(zip(["15", "30", "45", "60"], model_dm)),
         "paper_nM": dict(zip(["15", "30", "45", "60"], f4f["dm_BR_nM"])),
         "relative_error": dict(zip(["15", "30", "45", "60"], rel_all)),
         "model_ne_BR_60_nM": ne_60, "paper_ne_floor_nM": f4f["ne_floor_BR_nM"]},
        all(abs(e) <= 0.25 for e in rel_all), kind="paper_figure"))
    flank = {t: float(np.interp(60.0, s, frames[t])) for t in (30.0, 45.0, 60.0)}
    flank_peak = max(flank.values())
    rows.append(_row(
        "V1", "BR begins broad and low and contracts in time: the DM rises while lateral signal is lost",
        "Umulis 2010 Fig. 4E-F, p.266; O'Connor et al. 2006 ('rapid loss of pMad signal from nearby lateral cells')",
        "DM BR and FWHM at 30 vs 60 min, and BR 60 um from the DM (~12 nuclei): must fall after its peak",
        {"width30_um": w30, "width60_um": w60, "peak30_nM": r30["peak"], "peak60_nM": r60["peak"],
         "flank60um_BR_nM": {str(k): v for k, v in flank.items()}},
        w30 > w60 and r60["peak"] > r30["peak"] and flank[60.0] < flank_peak))
    fwhm_pct = r60["fwhm_percent_circumference"]
    rows.append(_row(
        "V2", "High BR is a narrow stripe at the dorsal midline (~10% of the circumference)", "Umulis 2010 p.260",
        "60-min FWHM as % of circumference (pass band 4-25%) and peak position",
        {"fwhm_percent_circumference": fwhm_pct, "peak_location_um_from_dm": r60["peak_location_um_from_dm"],
         "paper_percent": 10.0},
        4.0 <= fwhm_pct <= 25.0 and r60["peak_location_um_from_dm"] <= 5.0))
    sog = simulate_cross_section(perturbation="sog_het", save_times=[60.0])
    bsog = np.asarray(sog["fields"]["BR"][-1])
    rsog = _profile_readout(s, bsog, reference_max=r60["peak"])
    rwt = _profile_readout(s, frames[60.0], reference_max=r60["peak"])
    d20 = rsog["widths_reference_max"]["0.2"]["cells_5um"] - rwt["widths_reference_max"]["0.2"]["cells_5um"]
    d40 = rsog["widths_reference_max"]["0.4"]["cells_5um"] - rwt["widths_reference_max"]["0.4"]["cells_5um"]
    v3_measured = {"wt_width20_cells": rwt["widths_reference_max"]["0.2"]["cells_5um"],
                   "sog_het_width20_cells": rsog["widths_reference_max"]["0.2"]["cells_5um"],
                   "difference20_cells": d20, "difference40_cells": d40, "paper_difference20_cells": [2.0, 4.0]}
    rows.append(_row(
        "V3", "sog+/- is wider than wt at T = 0.2 and closer to wt at T = 0.4 (direction)", "Umulis 2010 p.265, Fig. 3E-F",
        "Full widths at 0.2 and 0.4 of the wt maximum, in 5-um cells", v3_measured, d20 > 0.0 and d40 < d20))
    rows.append(_row(
        "V3b", "sog+/- is 2-4 nuclei wider than wt at T = 0.2 (magnitude)", "Umulis 2010 p.265",
        "Width difference at 0.2 of the wt maximum (pass: 1-8 cells, i.e. within 2x of the measured range)",
        v3_measured, 1.0 <= d20 <= 8.0))
    tld = simulate_cross_section(perturbation="tld_null", save_times=[60.0])
    tldpeak = float(np.max(tld["fields"]["BR"][-1]))
    rows.append(_row(
        "V4", "tld-/- has essentially no signaling", "Umulis 2006 Fig. 3a",
        "Mutant peak BR / wt peak BR (pass < 0.2)", {"wt_peak_nM": r60["peak"], "tld_null_peak_nM": tldpeak, "peak_ratio": tldpeak / r60["peak"]},
        tldpeak / r60["peak"] < 0.2))
    sn = simulate_cross_section(perturbation="sog_null", save_times=[60.0])
    bsn = np.asarray(sn["fields"]["BR"][-1])
    rsn = _profile_readout(s, bsn, reference_max=r60["peak"])
    rows.append(_row(
        "V5", "sog-/- signaling is broader but weaker at the midline", "O'Connor et al. 2006 Development (review)",
        "DM BR and width at 0.2 of the wt maximum",
        {"wt_dm_nM": float(frames[60.0][0]), "sog_null_dm_nM": float(bsn[0]),
         "wt_width20_um": rwt["widths_reference_max"]["0.2"]["um"], "sog_null_width20_um": rsn["widths_reference_max"]["0.2"]["um"]},
        bsn[0] < frames[60.0][0] and rsn["widths_reference_max"]["0.2"]["um"] > rwt["widths_reference_max"]["0.2"]["um"]))
    iv = simulate_cross_section(perturbation="invitro_k3", save_times=[60.0])
    biv = np.asarray(iv["fields"]["BR"][-1])
    riv = _profile_readout(s, biv, circumference=circ)
    lost = riv["widths_own_max"]["0.5"]["um"] > 2.0 * w60 or riv["peak_location_um_from_dm"] > 5.0 or biv[0] < 0.3 * frames[60.0][0]
    rows.append(_row(
        "V6", "With the in-vitro Sog/BMP on-rate the pattern no longer matches (localisation lost)", "Umulis 2010 p.267-268, Fig. 5B",
        "k3 = 0.0168 nM^-1 min^-1: FWHM ratio, peak position, DM ratio to wt",
        {"wt_fwhm_um": w60, "invitro_fwhm_um": riv["widths_own_max"]["0.5"]["um"],
         "fwhm_ratio": riv["widths_own_max"]["0.5"]["um"] / max(w60, 1e-30),
         "peak_location_um_from_dm": riv["peak_location_um_from_dm"], "dm_ratio_to_wt": float(biv[0] / frames[60.0][0])},
        lost))
    nofb = simulate_cross_section(overrides={"Lambda": 0.0}, save_times=[30.0, 45.0, 60.0])
    nf_frames = {t: np.asarray(nofb["fields"]["BR"][nofb["times"].index(t)]) for t in (30.0, 45.0, 60.0)}
    nf_flank = {t: float(np.interp(60.0, s, nf_frames[t])) for t in (30.0, 45.0, 60.0)}
    rows.append(_row(
        "V7", "Without positive feedback lateral signal is not attenuated and the DM accumulates less",
        "Umulis 2010 p.260-261 ('a loss of ... positive feedback impedes the attenuation of pMad laterally as well as "
        "the accumulation of pMad signaling at the dorsal midline')",
        "Same Table S8 set with the SBP feedback removed (Lambda = 0): 60-min DM BR and BR 60 um from the DM vs wt",
        {"wt_dm_60_nM": float(frames[60.0][0]), "no_feedback_dm_60_nM": float(nf_frames[60.0][0]),
         "wt_flank60um_60_nM": flank[60.0], "no_feedback_flank60um_60_nM": nf_flank[60.0],
         "no_feedback_flank60um_nM": {str(k): v for k, v in nf_flank.items()}},
        nf_frames[60.0][0] < frames[60.0][0] and nf_flank[60.0] > flank[60.0]))
    nf = simulate_cross_section(mechanism="none", save_times=[30.0, 60.0])
    rn60 = _profile_readout(s, np.asarray(nf["fields"]["BR"][-1]), circumference=circ)
    rows.append(_row(
        "M1", "The fitted no-feedback set (Table S2) also forms a dorsal stripe with the shared Sog source",
        "Umulis 2010 Fig. 4B (every mechanism fits the wt data comparably, RMSD/mu 0.35-0.40)",
        "Table S2 parameters with the default Sog source: 60-min FWHM as % of circumference (pass < 50%)",
        {"fwhm_percent_circumference": rn60["fwhm_percent_circumference"], "peak_BR_nM": rn60["peak"],
         "explanation": "Table S2's receptor level (2386 nM) and Tld rate were fitted against the unpublished FISH Sog field; "
                        "with the 2006 Sog secretion rate this set stays flat (it localises only for phi_S <~ 300 nM/min)."},
        rn60["fwhm_percent_circumference"] < 50.0, kind="reproduction_gap"))

    if include_surface:
        grid = dict(nu=48, nv=32)
        s400 = simulate_surface(save_times=[60.0], **grid)
        s750 = simulate_surface(length_ap=750.0, save_times=[60.0], **grid)
        s750c = simulate_surface(length_ap=750.0, conserve=True, save_times=[60.0], **grid)
        mid = grid["nu"] // 2

        def norm(v): return v / max(float(np.max(v)), 1e-30)
        p400 = norm(np.asarray(s400["fields"]["BR"][-1])[mid])
        p750 = norm(np.asarray(s750["fields"]["BR"][-1])[mid])
        p750c = norm(np.asarray(s750c["fields"]["BR"][-1])[mid])
        rms_unc = float(np.sqrt(np.mean((p750 - p400) ** 2)))
        rms_con = float(np.sqrt(np.mean((p750c - p400) ** 2)))
        splits750 = [float(f) for f, yes in zip(s750["readouts"]["dorsal_midline"]["ap_fraction"], s750["readouts"]["split_map_vs_ap"]) if yes]
        splits400 = [float(f) for f, yes in zip(s400["readouts"]["dorsal_midline"]["ap_fraction"], s400["readouts"]["split_map_vs_ap"]) if yes]
        label = f"{grid['nu']}x{grid['nv']}"
        rows.append(_row(
            "V8", "A 750-um embryo with fixed production/receptor concentrations (shuttling only) splits into two stripes",
            "Umulis 2010 Fig. 7, p.271-272 ('splits at ~25% embryo length ... two parallel stripes')",
            f"{label} surface (the same answer at 32x24 and 64x40): AP positions where the 60-min BR maximum leaves the DM",
            {"grid": label, "split_ap_fractions_400": splits400, "split_ap_fractions_750": splits750,
             "note": "the paper used the reconstructed (non-symmetric) VirtualEmbryo geometry and FISH prepatterns; "
                     "this model uses a symmetric prolate spheroid"},
            bool(splits750) and not splits400))
        dmc = s750c["readouts"]["dorsal_midline"]
        dm_c = np.asarray(dmc["BR"])
        pole_ratio = float(min(dm_c[0], dm_c[-1]) / max(float(np.max(dm_c)), 1e-30))
        rows.append(_row(
            "V8b", "With the conservation conditions a large embryo loses signal at the poles and its maximum moves to ~1/2 EL",
            "Umulis 2010 p.272 (575-750 um predictions; consistent with D. virilis pMad)",
            f"{label} surface, 750 um, Supp. Eqs 95-104: pole/max DM BR ratio (< 0.5) and AP position of the maximum (0.4-0.6); "
            "plus RMS deviation of the AP-midline DV profile from 400 um with and without conservation",
            {"grid": label, "pole_to_max_ratio": pole_ratio, "peak_ap_fraction": dmc["peak_ap_fraction"],
             "rms_750_unconserved": rms_unc, "rms_750_conserved": rms_con},
            pole_ratio < 0.5 and 0.4 <= dmc["peak_ap_fraction"] <= 0.6))
        dm = s400["readouts"]["dorsal_midline"]
        rows.append(_row(
            "V9", "On the embryo surface, dorsal-midline BR varies along the AP axis", "Umulis 2010 Fig. 4E, 6, p.264",
            "Min/max ratio of 60-min DM BR along AP and the AP position of the maximum",
            {"grid": label, "dm_min_max_ratio": dm["min_max_ratio"], "peak_ap_fraction": dm["peak_ap_fraction"],
             "note": "the maximum sits at the termini, where Fig. 4A places dpp around the whole circumference"},
            dm["min_max_ratio"] < 0.95))
    for_pass = [r for r in rows if r["pass"] is not None]
    required = [r for r in rows if r["id"] in ("S1", "D1", "V1", "V2", "V4", "V6", "V7")]
    return {"citation": CITATION, "citation_2006": CITATION_2006, "include_surface": bool(include_surface),
            "parameter_set": DEFAULT_PARAMETER_SET, "checks": rows,
            "passed": sum(bool(r["pass"]) for r in for_pass), "total": len(for_pass),
            "required_ids": [r["id"] for r in required],
            "required_pass": all(r["pass"] for r in required),
            "wall_time_s": time.perf_counter() - started, "assumptions": list(ASSUMPTIONS),
            "paper_data": PAPER_DATA}


def _round_tree(obj: Any, ndigits: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [_round_tree(x, ndigits) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_tree(v, ndigits) for k, v in obj.items()}
    return obj


def export_cache(path: str | Path = "static/data/bmp/cache.json", nu: int = 40, nv: int = 24) -> dict[str, Any]:
    """Precompute what the browser shows: surface BR movies and cross-section time courses."""
    started = time.perf_counter()
    surface_times = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    surface = {}
    grid_meta = None
    for pert in ("wt", "sog_het", "sog_null", "tld_null", "invitro_k3"):
        res = simulate_surface(perturbation=pert, nu=nu, nv=nv, save_times=surface_times)
        if grid_meta is None:
            grid_meta = {k: res["grid"][k] for k in ("nu", "nv", "a_um", "b_um", "u", "v")}
        surface[pert] = {"times": res["times"], "BR": res["fields"]["BR"],
                         "dorsal_midline": res["readouts"]["dorsal_midline"],
                         "wall_time_s": res["diagnostics"]["wall_time_s"]}
    cross = {}
    cs_times = [float(t) for t in range(0, 61, 5)]
    for mech in MECHANISMS:
        for pert in PERTURBATIONS:
            if mech != "sbp" and pert not in ("wt", "sog_het", "tld_null"):
                continue
            res = simulate_cross_section(mechanism=mech, perturbation=pert, save_times=cs_times)
            cross[f"{mech}/{pert}"] = {"times": res["times"], "s_um": res["grid"]["s_um"],
                                       "BR": res["fields"]["BR"], "B": res["fields"]["B"],
                                       "S": res["fields"]["S"], "IB": res["fields"]["IB"],
                                       "sources": res["sources"]}
    as_printed = simulate_cross_section(parameter_set="umulis2010_as_printed", save_times=cs_times)
    cross["sbp/wt@as_printed"] = {"times": as_printed["times"], "s_um": as_printed["grid"]["s_um"],
                                  "BR": as_printed["fields"]["BR"], "B": as_printed["fields"]["B"],
                                  "S": as_printed["fields"]["S"], "IB": as_printed["fields"]["IB"],
                                  "sources": as_printed["sources"]}
    r06 = simulate_cross_section(parameter_set="umulis2006", t_end=600.0,
                                 save_times=[15.0, 30.0, 45.0, 60.0, 600.0])
    cross["umulis2006/wt"] = {"times": r06["times"], "s_um": r06["grid"]["s_um"], "BR": r06["fields"]["BR"],
                              "B": r06["fields"]["B"]}
    payload = {"schema_version": 2, "citation": CITATION, "citation_2006": CITATION_2006,
               "parameter_set": DEFAULT_PARAMETER_SET, "surface_grid": grid_meta, "surface": surface,
               "cross_section": cross, "paper_data": PAPER_DATA, "perturbations": PERTURBATIONS,
               "assumptions": ASSUMPTIONS, "precision": "4 decimals from float64 BDF output"}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_round_tree(payload), separators=(",", ":"), allow_nan=False)
    target.write_text(text, encoding="utf-8")
    size = target.stat().st_size
    if size >= 8 * 1024 * 1024:
        raise RuntimeError(f"cache is {size} bytes, exceeding 8 MiB")
    return {"path": str(target), "bytes": size, "surface_runs": list(surface), "cross_section_runs": list(cross),
            "wall_time_s": time.perf_counter() - started}


__all__ = ["MECHANISMS", "PARAMETER_SETS", "PERTURBATIONS", "PARAMETER_TABLE", "ASSUMPTIONS", "CITATION",
           "PAPER_DATA", "model_info", "simulate_cross_section", "simulate_surface", "grid_convergence",
           "validate", "export_cache"]
