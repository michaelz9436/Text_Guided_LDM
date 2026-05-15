"""
Molecular descriptor generator with diverse natural-language output.

Design for contrastive learning: same molecule descriptions must be
more similar to each other than to descriptions of different molecules.

Key strategies:
  - "Identity phrases" (atom composition, ring signature, specific groups)
    are ALWAYS included and use mild paraphrase only.
  - "Detail phrases" (polarity level, flexibility, hydrophobicity)
    get heavier paraphrase, dropout, and shuffle — these add diversity
    without destroying the molecular fingerprint.
  - Order is shuffled per variant.
  - Sentence-level templates vary the wrapper style.
"""

import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

# ═══════════════════════════════════════════════════════════════════════════
# SMARTS
# ═══════════════════════════════════════════════════════════════════════════

FUNCTIONAL_GROUPS: Dict[str, str] = {
    "hydroxyl":    "[OX2H]",
    "carboxyl":    "[CX3](=O)[OX2H1]",
    "primary amine":"[NX3;H2;!$(NC=O)]",
    "secondary amine":"[NX3;H1;!$(NC=O)]",
    "amide":       "[NX3][CX3](=[OX1])[#6]",
    "ester":       "[#6][CX3](=O)[OX2H0][#6]",
    "ether":       "[OD2]([#6])[#6]",
    "aldehyde":    "[CX3H1](=O)[#6]",
    "ketone":      "[#6][CX3](=O)[#6]",
    "nitro":       "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "nitrile":     "[#6]C#N",
    "sulfonyl":    "[SX4](=O)(=O)",
    "thiol":       "[SX2H]",
    "thioether":   "[#6][SX2][#6]",
    "phosphate":   "[PX4](=O)([OX2])[OX2]",
    "fluorine":    "[F]",
    "chlorine":    "[Cl]",
    "bromine":     "[Br]",
    "iodine":      "[I]",
    "epoxide":     "C1OC1",
    "acyl halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "anhydride":   "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    "imine":       "[CX3;$([C]([#6])[#6])]=N",
    "azo":         "[#6]N=N[#6]",
}

RING_SMARTS: Dict[str, str] = {
    "benzene":     "c1ccccc1",
    "pyridine":    "c1ccncc1",
    "pyrrole":     "c1cc[nH]c1",
    "furan":       "c1ccoc1",
    "thiophene":   "c1ccsc1",
    "imidazole":   "c1cnc[nH]1",
    "pyrimidine":  "c1ccnc(n1)",
    "naphthalene": "c1ccc2ccccc2c1",
    "cyclohexane": "C1CCCCC1",
    "cyclopentane":"C1CCCC1",
}

ELEMENT_SYMBOLS = {6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S",
                   17: "Cl", 35: "Br", 53: "I"}


def _count(mol: Chem.Mol, smarts: str) -> int:
    pat = Chem.MolFromSmarts(smarts)
    return len(mol.GetSubstructMatches(pat)) if pat else 0


# ═══════════════════════════════════════════════════════════════════════════
# Raw property extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract(mol: Chem.Mol) -> Dict:
    p: Dict = {}

    # atom composition (highly discriminative)
    atom_counts: Dict[int, int] = Counter()
    for atom in mol.GetAtoms():
        atom_counts[atom.GetAtomicNum()] += 1
    p["atom_comp"] = dict(atom_counts)
    p["n_heavy"] = mol.GetNumHeavyAtoms()

    # functional groups
    fg = {}
    for key, sma in FUNCTIONAL_GROUPS.items():
        c = _count(mol, sma)
        if c > 0:
            fg[key] = c
    p["fg"] = fg

    # rings
    p["n_aromatic"] = rdMolDescriptors.CalcNumAromaticRings(mol)
    p["n_aliphatic"] = rdMolDescriptors.CalcNumAliphaticRings(mol)
    p["n_rings"] = rdMolDescriptors.CalcNumRings(mol)
    ring_hits = {}
    for key, sma in RING_SMARTS.items():
        c = _count(mol, sma)
        if c > 0:
            ring_hits[key] = c
    p["ring_types"] = ring_hits

    # properties
    p["tpsa"] = Descriptors.TPSA(mol)
    p["hbd"] = Lipinski.NumHDonors(mol)
    p["hba"] = Lipinski.NumHAcceptors(mol)
    p["logp"] = Descriptors.MolLogP(mol)
    p["mw"] = Descriptors.ExactMolWt(mol)
    p["rot_bonds"] = Lipinski.NumRotatableBonds(mol)
    p["frac_sp3"] = Descriptors.FractionCSP3(mol)
    p["charge"] = Chem.GetFormalCharge(mol)
    return p


# ═══════════════════════════════════════════════════════════════════════════
# Phrase generators
#
# Two tiers:
#   IDENTITY  – always kept, mild paraphrase (atom comp, FGs, rings)
#   DETAIL    – subject to dropout, heavier paraphrase
# ═══════════════════════════════════════════════════════════════════════════

# ---- identity tier ----

def _ph_atom_composition(p: Dict, rng: random.Random) -> List[str]:
    """Atom formula — highly unique per molecule."""
    comp = p["atom_comp"]
    n_heavy = p["n_heavy"]

    parts = []
    for z in sorted(comp.keys()):
        sym = ELEMENT_SYMBOLS.get(z, f"Z{z}")
        parts.append(f"{comp[z]}{sym}")
    formula = " ".join(parts)

    templates = [
        f"composed of {formula}",
        f"molecular formula {formula}",
        f"atom composition: {formula}",
        f"contains {formula}",
        f"made up of {formula}",
    ]
    base = rng.choice(templates)

    if rng.random() < 0.5:
        base += rng.choice([
            f" ({n_heavy} heavy atoms)",
            f", {n_heavy} heavy atoms total",
            f", totaling {n_heavy} heavy atoms",
        ])
    return [base]


def _ph_functional_groups(p: Dict, rng: random.Random) -> List[str]:
    """Functional groups — use consistent naming, mild variation."""
    out = []
    # mild paraphrase: only vary the verb, keep the group name stable
    verbs_s = ["contains", "has", "bears", "with", "featuring"]
    verbs_p = ["contains", "has", "with", "bearing"]

    for key, cnt in p["fg"].items():
        name = key  # use the canonical name directly
        if cnt == 1:
            out.append(f"{rng.choice(verbs_s)} {name}")
        else:
            out.append(f"{rng.choice(verbs_p)} {cnt} {name}s")
    return out


def _ph_rings(p: Dict, rng: random.Random) -> List[str]:
    out = []
    na, nal = p["n_aromatic"], p["n_aliphatic"]

    if na == 0:
        out.append(rng.choice(["no aromatic rings", "non-aromatic",
                                "lacks aromatic rings"]))
    else:
        s = "s" if na > 1 else ""
        out.append(rng.choice([
            f"{na} aromatic ring{s}",
            f"has {na} aromatic ring{s}",
            f"contains {na} aromatic ring{s}",
        ]))

    if nal > 0:
        s = "s" if nal > 1 else ""
        out.append(rng.choice([
            f"{nal} aliphatic ring{s}",
            f"has {nal} aliphatic ring{s}",
        ]))

    for key, cnt in p["ring_types"].items():
        if cnt == 1:
            out.append(rng.choice([f"{key} ring", f"a {key} ring",
                                    f"contains {key} ring"]))
        else:
            out.append(f"{cnt} {key} rings")
    return out


# ---- detail tier ----

def _ph_polarity(p: Dict, rng: random.Random) -> List[str]:
    out = []
    tpsa = p["tpsa"]
    if tpsa > 90:
        out.append(rng.choice(["high polarity", "highly polar",
                                "very polar"]))
    elif tpsa > 40:
        out.append(rng.choice(["moderate polarity", "moderately polar",
                                "intermediate polarity"]))
    else:
        out.append(rng.choice(["low polarity", "weakly polar",
                                "relatively nonpolar"]))

    hbd, hba = p["hbd"], p["hba"]
    if hbd > 0 and hba > 0 and rng.random() < 0.5:
        out.append(f"{hbd} H-bond donors and {hba} H-bond acceptors")
    else:
        if hbd > 0:
            out.append(f"{hbd} H-bond donor{'s' if hbd>1 else ''}")
        if hba > 0:
            out.append(f"{hba} H-bond acceptor{'s' if hba>1 else ''}")
    return out


def _ph_hydrophobicity(p: Dict, rng: random.Random) -> List[str]:
    logp = p["logp"]
    v = f"{logp:.1f}"

    if logp > 3.0:
        desc = rng.choice(["highly hydrophobic", "very lipophilic",
                           "strongly hydrophobic"])
    elif logp > 1.0:
        desc = rng.choice(["moderately hydrophobic", "somewhat lipophilic",
                           "mildly hydrophobic"])
    elif logp > -1.0:
        desc = rng.choice(["balanced hydrophilicity",
                           "intermediate hydrophilicity"])
    else:
        desc = rng.choice(["hydrophilic", "strongly hydrophilic",
                           "water-soluble"])

    if rng.random() < 0.6:
        return [f"{desc} (logP={v})"]
    return [f"logP={v}, {desc}"]


def _ph_size(p: Dict, rng: random.Random) -> List[str]:
    mw = p["mw"]
    if mw > 500:
        sz = rng.choice(["large molecule", "high molecular weight"])
    elif mw > 200:
        sz = rng.choice(["medium-sized molecule", "moderate molecular weight"])
    else:
        sz = rng.choice(["small molecule", "low molecular weight"])

    r = rng.random()
    if r < 0.33:
        return [f"{sz} (MW={mw:.1f})"]
    elif r < 0.66:
        return [f"{sz}, MW={mw:.0f}"]
    return [sz]


def _ph_flexibility(p: Dict, rng: random.Random) -> List[str]:
    rot = p["rot_bonds"]
    if rot == 0:
        return [rng.choice(["rigid structure", "no rotatable bonds",
                            "conformationally rigid"])]
    s = "s" if rot > 1 else ""
    if rot <= 3:
        return [rng.choice([
            f"low flexibility ({rot} rotatable bond{s})",
            f"relatively rigid ({rot} rotatable bond{s})",
        ])]
    elif rot <= 7:
        return [rng.choice([
            f"moderate flexibility ({rot} rotatable bonds)",
            f"moderately flexible ({rot} rotatable bonds)",
        ])]
    else:
        return [rng.choice([
            f"highly flexible ({rot} rotatable bonds)",
            f"very flexible ({rot} rotatable bonds)",
        ])]


def _ph_saturation(p: Dict, rng: random.Random) -> List[str]:
    f3 = p["frac_sp3"]
    if f3 > 0.8:
        return [rng.choice(["highly saturated", "predominantly saturated"])]
    elif f3 > 0.4:
        return [rng.choice(["partially saturated", "mixed saturation"])]
    else:
        return [rng.choice(["highly unsaturated", "predominantly unsaturated"])]


def _ph_charge(p: Dict, rng: random.Random) -> List[str]:
    q = p["charge"]
    if q > 0:
        return [rng.choice([f"positively charged (+{q})",
                            f"cationic (charge +{q})"])]
    elif q < 0:
        return [rng.choice([f"negatively charged ({q})",
                            f"anionic (charge {q})"])]
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Generator registry — separate identity vs detail
# ═══════════════════════════════════════════════════════════════════════════

IDENTITY_GENERATORS = [
    _ph_atom_composition,
    _ph_functional_groups,
    _ph_rings,
]

DETAIL_GENERATORS = [
    _ph_polarity,
    _ph_hydrophobicity,
    _ph_size,
    _ph_flexibility,
    _ph_saturation,
    _ph_charge,
]


# ═══════════════════════════════════════════════════════════════════════════
# Sentence assembly
# ═══════════════════════════════════════════════════════════════════════════

def _join_and(parts: List[str], rng: random.Random) -> str:
    if len(parts) <= 1:
        return parts[0] if parts else ""
    if len(parts) == 2:
        return parts[0] + rng.choice([" and ", " and also "]) + parts[1]
    return ", ".join(parts[:-1]) + rng.choice([", and ", ", "]) + parts[-1]


def _assemble(parts: List[str], rng: random.Random) -> str:
    style = rng.randint(0, 3)
    if style == 0:
        return ", ".join(parts)
    elif style == 1:
        return "a molecule with " + _join_and(parts, rng)
    elif style == 2:
        verb = rng.choice(["features", "exhibits", "shows", "displays",
                           "is characterized by"])
        subj = rng.choice(["this compound", "the molecule", "this molecule"])
        return f"{subj} {verb} " + _join_and(parts, rng)
    else:
        return "; ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════════════════════

def _parse_mol(mol_input: Union[str, Chem.Mol]) -> Chem.Mol:
    if isinstance(mol_input, Chem.Mol):
        return mol_input
    mol = Chem.MolFromSmiles(mol_input)
    if mol is None:
        mol = Chem.MolFromMolBlock(mol_input)
    if mol is None:
        raise ValueError(f"Cannot parse molecule: {mol_input!r}")
    return mol


def generate_descriptor(
    mol_input: Union[str, Chem.Mol],
    n_variants: int = 1,
    dropout_rate: float = 0.0,
    seed: Optional[int] = None,
) -> Union[str, List[str]]:
    """
    Generate natural-language molecular descriptions.

    Identity features (atom composition, functional groups, ring types)
    are ALWAYS included. Detail features (polarity, size, flexibility …)
    are subject to dropout.

    Returns single string if n_variants=1, else list of strings.
    """
    mol = _parse_mol(mol_input)
    props = _extract(mol)
    rng = random.Random(seed)

    results = []
    for _ in range(n_variants):
        # --- identity phrases: always kept ---
        identity_phrases: List[str] = []
        id_sections: List[List[str]] = []
        for gen_fn in IDENTITY_GENERATORS:
            phr = gen_fn(props, rng)
            if phr:
                id_sections.append(phr)
        rng.shuffle(id_sections)
        for sec in id_sections:
            identity_phrases.extend(sec)

        # --- detail phrases: subject to dropout ---
        detail_phrases: List[str] = []
        det_sections: List[List[str]] = []
        for gen_fn in DETAIL_GENERATORS:
            phr = gen_fn(props, rng)
            if phr:
                det_sections.append(phr)
        rng.shuffle(det_sections)

        for sec in det_sections:
            for phrase in sec:
                if dropout_rate <= 0 or rng.random() >= dropout_rate:
                    detail_phrases.append(phrase)

        # guarantee at least 1 detail phrase
        if not detail_phrases and det_sections:
            all_det = [p for sec in det_sections for p in sec]
            if all_det:
                detail_phrases.append(rng.choice(all_det))

        # merge and shuffle the combined list
        all_phrases = identity_phrases + detail_phrases
        rng.shuffle(all_phrases)

        results.append(_assemble(all_phrases, rng))

    return results[0] if n_variants == 1 else results


def generate_descriptor_from_sdf(sdf_path: str, mol_index: int = 0, **kw) -> str:
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)
    mols = [m for m in supplier if m is not None]
    if not mols:
        raise ValueError(f"No valid molecules in {sdf_path}")
    if mol_index >= len(mols):
        raise IndexError(f"mol_index={mol_index} but file has {len(mols)} molecules")
    return generate_descriptor(mols[mol_index], **kw)


# ═══════════════════════════════════════════════════════════════════════════
# Batch processing
# ═══════════════════════════════════════════════════════════════════════════

def batch_generate(
    sdf_dir: str,
    output_csv: str,
    n_variants: int = 10,
    dropout_rate: float = 0.3,
    num_display: int = 3,
    seed: int = 42,
) -> None:
    import csv, glob

    sdf_files = sorted(glob.glob(os.path.join(sdf_dir, "*.sdf")))
    print(f"Found {len(sdf_files)} SDF files in {sdf_dir}")
    print(f"Generating {n_variants} variants/mol  dropout={dropout_rate}\n")

    header = ["filename"] + [f"desc_{i}" for i in range(n_variants)]
    rows = []

    for i, path in enumerate(sdf_files):
        fname = os.path.basename(path)
        try:
            supplier = Chem.SDMolSupplier(path, removeHs=True)
            mol = next((m for m in supplier if m is not None), None)
            if mol is None:
                rows.append([fname] + ["FAILED"] * n_variants)
                continue
            descs = generate_descriptor(
                mol, n_variants=n_variants,
                dropout_rate=dropout_rate, seed=seed + i,
            )
            rows.append([fname] + descs)
            if i < num_display:
                print(f"[{fname}]")
                for j, d in enumerate(descs):
                    print(f"  {j:>2}: {d}")
                print()
        except Exception as e:
            rows.append([fname] + [f"ERROR: {e}"] * n_variants)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    ok = sum(1 for r in rows if r[1] not in ("FAILED",)
             and not r[1].startswith("ERROR"))
    print(f"{'='*60}")
    print(f"Done!  {ok}/{len(rows)} molecules OK.")
    print(f"Saved to: {output_csv}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Molecular descriptor generator")
    ap.add_argument("--sdf_dir",  type=str, default=None)
    ap.add_argument("--output",   type=str, default="descriptor_output.csv")
    ap.add_argument("--smiles",   type=str, default=None)
    ap.add_argument("--sdf",      type=str, default=None)
    ap.add_argument("--n_variants", type=int, default=10)
    ap.add_argument("--dropout",  type=float, default=0.3)
    ap.add_argument("--num_display", type=int, default=3)
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    if args.smiles:
        for i, d in enumerate(generate_descriptor(
            args.smiles, n_variants=args.n_variants,
            dropout_rate=args.dropout, seed=args.seed,
        )):
            print(f"  {i}: {d}")
    elif args.sdf:
        print(generate_descriptor_from_sdf(args.sdf))
    elif args.sdf_dir:
        batch_generate(
            args.sdf_dir, args.output,
            n_variants=args.n_variants, dropout_rate=args.dropout,
            num_display=args.num_display, seed=args.seed,
        )
    else:
        print("Demo: Aspirin — 10 diverse descriptions\n")
        for i, d in enumerate(generate_descriptor(
            "CC(=O)Oc1ccccc1C(=O)O",
            n_variants=10, dropout_rate=0.3, seed=0,
        )):
            print(f"  {i}: {d}")
