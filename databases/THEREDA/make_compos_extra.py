"""Generate data-in/Compos.extra.json: more GEMS 'compos' (composition) records for THEREDA.

Each record is 1 mole of a salt / mineral / gas / acid, written in the same form as the hand-made
data-in/Compos.backup.json (elements as independent components, amounts in moles, molar mass from the
THEREDA IComp records), so that an input composition can be defined from the elements of THEREDA.
Only elements that exist in the exported IComp (data-out/<dir>/IComp.backup.json) are used.

The export notebook merges Compos.backup.json and Compos.extra.json (keys already in Compos.backup.json win)
and converts them to compos.THEREDA.<version>.pdb/.ndx.  Run: python make_compos_extra.py [IComp.backup.json]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from gems3_split import composition  # noqa: E402

# (key name <= 13 chars, formula, type AQ|MIN|GA, long name <= 27 chars, description)
SPECIES = [
    # ---- aqueous: electrolytes, acids, bases (1 mol dissolved) -----------------------------------------
    ("NaHCO3", "NaHCO3", "AQ", "Sodium-bicarbonate_1M_", "NaHCO3 sodium hydrogen carbonate"),
    ("K2CO3", "K2CO3", "AQ", "Potassium-carbonate_1M_", "K2CO3 potassium carbonate"),
    ("KHCO3", "KHCO3", "AQ", "Potassium-bicarbonate_1M_", "KHCO3 potassium hydrogen carbonate"),
    ("KNO3", "KNO3", "AQ", "Potassium-nitrate_1M_", "KNO3 potassium nitrate"),
    ("NaNO2", "NaNO2", "AQ", "Sodium-nitrite_1M_", "NaNO2 sodium nitrite"),
    ("Mg(NO3)2", "Mg(NO3)2", "AQ", "Magnesium-nitrate_1M_", "Mg(NO3)2 magnesium nitrate"),
    ("Sr(NO3)2", "Sr(NO3)2", "AQ", "Strontium-nitrate_1M_", "Sr(NO3)2 strontium nitrate"),
    ("SrCl2", "SrCl2", "AQ", "Strontium-chloride_1M_", "SrCl2 strontium chloride"),
    ("Cs2SO4", "Cs2SO4", "AQ", "Cesium-sulfate_1M_", "Cs2SO4 cesium sulfate"),
    ("CsNO3", "CsNO3", "AQ", "Cesium-nitrate_1M_", "CsNO3 cesium nitrate"),
    ("ZnCl2", "ZnCl2", "AQ", "Zinc-chloride_1M_", "ZnCl2 zinc chloride"),
    ("ZnSO4", "ZnSO4", "AQ", "Zinc-sulfate_1M_", "ZnSO4 zinc sulfate"),
    ("Zn(NO3)2", "Zn(NO3)2", "AQ", "Zinc-nitrate_1M_", "Zn(NO3)2 zinc nitrate"),
    ("CdCl2", "CdCl2", "AQ", "Cadmium-chloride_1M_", "CdCl2 cadmium chloride"),
    ("CdSO4", "CdSO4", "AQ", "Cadmium-sulfate_1M_", "CdSO4 cadmium sulfate"),
    ("Cd(NO3)2", "Cd(NO3)2", "AQ", "Cadmium-nitrate_1M_", "Cd(NO3)2 cadmium nitrate"),
    ("Pb(NO3)2", "Pb(NO3)2", "AQ", "Lead-nitrate_1M_", "Pb(NO3)2 lead nitrate"),
    ("NH4Cl", "NH4Cl", "AQ", "Ammonium-chloride_1M_", "NH4Cl ammonium chloride"),
    ("NH4NO3", "NH4NO3", "AQ", "Ammonium-nitrate_1M_", "NH4NO3 ammonium nitrate"),
    ("(NH4)2SO4", "(NH4)2SO4", "AQ", "Ammonium-sulfate_1M_", "(NH4)2SO4 ammonium sulfate"),
    ("NH3", "NH3", "AQ", "Ammonia-aq_1M_", "NH3 ammonia, dissolved"),
    ("NaH2PO4", "NaH2PO4", "AQ", "Sodium-dihydrogenphosphate_", "NaH2PO4 sodium dihydrogen phosphate"),
    ("Na2HPO4", "Na2HPO4", "AQ", "Sodium-hydrogenphosphate_1M_", "Na2HPO4 disodium hydrogen phosphate"),
    ("Na3PO4", "Na3PO4", "AQ", "Sodium-phosphate_1M_", "Na3PO4 trisodium phosphate"),
    ("KH2PO4", "KH2PO4", "AQ", "Potassium-dihydrogenphosph_", "KH2PO4 potassium dihydrogen phosphate"),
    ("K2HPO4", "K2HPO4", "AQ", "Potassium-hydrogenphosphat_", "K2HPO4 dipotassium hydrogen phosphate"),
    ("Na2SeO4", "Na2SeO4", "AQ", "Sodium-selenate_1M_", "Na2SeO4 sodium selenate"),
    ("Na2SeO3", "Na2SeO3", "AQ", "Sodium-selenite_1M_", "Na2SeO3 sodium selenite"),
    ("Na2SiO3", "Na2SiO3", "AQ", "Sodium-silicate_1M_", "Na2SiO3 sodium metasilicate"),
    ("NaAl(OH)4", "NaAl(OH)4", "AQ", "Sodium-aluminate_1M_", "NaAl(OH)4 sodium aluminate"),
    ("AlCl3", "AlCl3", "AQ", "Aluminum-chloride_1M_", "AlCl3 aluminium chloride"),
    ("Al(NO3)3", "Al(NO3)3", "AQ", "Aluminum-nitrate_1M_", "Al(NO3)3 aluminium nitrate"),
    ("Al2(SO4)3", "Al2(SO4)3", "AQ", "Aluminum-sulfate_1M_", "Al2(SO4)3 aluminium sulfate"),
    ("Mg(ClO4)2", "Mg(ClO4)2", "AQ", "Magnesium-perchlorate_1M_", "Mg(ClO4)2 magnesium perchlorate"),
    ("Ca(ClO4)2", "Ca(ClO4)2", "AQ", "Calcium-perchlorate_1M_", "Ca(ClO4)2 calcium perchlorate"),
    ("KClO4", "KClO4", "AQ", "Potassium-perchlorate_1M_", "KClO4 potassium perchlorate"),
    ("Si(OH)4", "Si(OH)4", "AQ", "Silicic-acid_1M_", "Si(OH)4 silicic acid"),
    ("H2CO3", "H2CO3", "AQ", "Carbonic-acid_1M_", "H2CO3 carbonic acid"),
    ("Na2S", "Na2S", "AQ", "Sodium-sulfide_1M_", "Na2S sodium sulfide"),
    ("NaTcO4", "NaTcO4", "AQ", "Sodium-pertechnetate_1M_", "NaTcO4 sodium pertechnetate"),
    ("NdCl3", "NdCl3", "AQ", "Neodymium-chloride_1M_", "NdCl3 neodymium chloride"),
    ("Nd(NO3)3", "Nd(NO3)3", "AQ", "Neodymium-nitrate_1M_", "Nd(NO3)3 neodymium nitrate"),
    ("AmCl3", "AmCl3", "AQ", "Americium-chloride_1M_", "AmCl3 americium chloride"),
    ("Am(NO3)3", "Am(NO3)3", "AQ", "Americium-nitrate_1M_", "Am(NO3)3 americium nitrate"),
    ("Cm(NO3)3", "Cm(NO3)3", "AQ", "Curium-nitrate_1M_", "Cm(NO3)3 curium nitrate"),
    ("ThCl4", "ThCl4", "AQ", "Thorium-chloride_1M_", "ThCl4 thorium chloride"),
    ("Th(NO3)4", "Th(NO3)4", "AQ", "Thorium-nitrate_1M_", "Th(NO3)4 thorium nitrate"),
    ("UO2(NO3)2", "UO2(NO3)2", "AQ", "Uranyl-nitrate_1M_", "UO2(NO3)2 uranyl nitrate"),
    ("UO2Cl2", "UO2Cl2", "AQ", "Uranyl-chloride_1M_", "UO2Cl2 uranyl chloride"),
    ("UO2SO4", "UO2SO4", "AQ", "Uranyl-sulfate_1M_", "UO2SO4 uranyl sulfate"),
    # ---- solids: evaporite and other salt minerals, hydroxides, phosphates, carbonates -----------------
    ("Halite", "NaCl", "MIN", "Halite-NaCl_1M_", "NaCl halite"),
    ("Sylvite", "KCl", "MIN", "Sylvite-KCl_1M_", "KCl sylvite"),
    ("Carnallite", "KMgCl3(H2O)6", "MIN", "Carnallite_1M_", "KMgCl3(H2O)6 carnallite"),
    ("Bischofite", "MgCl2(H2O)6", "MIN", "Bischofite_1M_", "MgCl2(H2O)6 bischofite"),
    ("Antarcticite", "CaCl2(H2O)6", "MIN", "Antarcticite_1M_", "CaCl2(H2O)6 antarcticite"),
    ("Kieserite", "MgSO4(H2O)", "MIN", "Kieserite_1M_", "MgSO4(H2O) kieserite"),
    ("Epsomite", "MgSO4(H2O)7", "MIN", "Epsomite_1M_", "MgSO4(H2O)7 epsomite"),
    ("Glauberite", "Na2Ca(SO4)2", "MIN", "Glauberite_1M_", "Na2Ca(SO4)2 glauberite"),
    ("Polyhalite", "K2Ca2Mg(SO4)4(H2O)2", "MIN", "Polyhalite_1M_", "K2Ca2Mg(SO4)4(H2O)2 polyhalite"),
    ("Syngenite", "K2Ca(SO4)2(H2O)", "MIN", "Syngenite_1M_", "K2Ca(SO4)2(H2O) syngenite"),
    ("Mirabilite", "Na2SO4(H2O)10", "MIN", "Mirabilite_1M_", "Na2SO4(H2O)10 mirabilite"),
    ("Thenardite", "Na2SO4", "MIN", "Thenardite_1M_", "Na2SO4 thenardite"),
    ("Arcanite", "K2SO4", "MIN", "Arcanite_1M_", "K2SO4 arcanite"),
    ("Natron", "Na2CO3(H2O)10", "MIN", "Natron_1M_", "Na2CO3(H2O)10 natron"),
    ("Trona", "Na3H(CO3)2(H2O)2", "MIN", "Trona_1M_", "Na3H(CO3)2(H2O)2 trona"),
    ("Nahcolite", "NaHCO3", "MIN", "Nahcolite_1M_", "NaHCO3 nahcolite"),
    ("Hydroxylapat", "Ca5(PO4)3(OH)", "MIN", "Hydroxylapatite_1M_", "Ca5(PO4)3(OH) hydroxylapatite"),
    ("Ca3(PO4)2", "Ca3(PO4)2", "MIN", "Calcium-phosphate_1M_", "Ca3(PO4)2 tricalcium phosphate"),
    ("Brushite", "CaHPO4(H2O)2", "MIN", "Brushite_1M_", "CaHPO4(H2O)2 brushite"),
    ("Monetite", "CaHPO4", "MIN", "Monetite_1M_", "CaHPO4 monetite"),
    ("Cerussite", "PbCO3", "MIN", "Cerussite_1M_", "PbCO3 cerussite"),
    ("Anglesite", "PbSO4", "MIN", "Anglesite_1M_", "PbSO4 anglesite"),
    ("PbO", "PbO", "MIN", "Lead-oxide_1M_", "PbO lead oxide"),
    ("Smithsonite", "ZnCO3", "MIN", "Smithsonite_1M_", "ZnCO3 smithsonite"),
    ("Zincite", "ZnO", "MIN", "Zincite_1M_", "ZnO zincite"),
    ("Otavite", "CdCO3", "MIN", "Otavite_1M_", "CdCO3 otavite"),
    ("Boehmite", "AlOOH", "MIN", "Boehmite_1M_", "AlOOH boehmite"),
    ("Portlandite", "Ca(OH)2", "MIN", "Portlandite_1M_", "Ca(OH)2 portlandite"),
    ("Brucite", "Mg(OH)2", "MIN", "Brucite_1M_", "Mg(OH)2 brucite"),
    ("Hydrotalcite", "Mg4Al2(OH)14(H2O)3", "MIN", "Hydrotalcite_1M_", "Mg4Al2(OH)14(H2O)3 hydrotalcite"),
    ("Ettringite", "Ca6Al2(SO4)3(OH)12(H2O)26", "MIN", "Ettringite_1M_", "Ca6Al2(SO4)3(OH)12(H2O)26 ettringite"),
    ("Monosulfoalum", "Ca4Al2SO10(H2O)12", "MIN", "Monosulfoaluminate_1M_", "Ca4Al2SO10(H2O)12 monosulfoaluminate"),
    ("UO2", "UO2", "MIN", "Uraninite-UO2_1M_", "UO2 uraninite"),
    ("UO3", "UO3", "MIN", "Uranium-trioxide_1M_", "UO3 uranium trioxide"),
    ("Nd2O3", "Nd2O3", "MIN", "Neodymium-oxide_1M_", "Nd2O3 neodymium oxide"),
    ("Nd(OH)3", "Nd(OH)3", "MIN", "Neodymium-hydroxide_1M_", "Nd(OH)3 neodymium hydroxide"),
    ("AmO2", "AmO2", "MIN", "Americium-dioxide_1M_", "AmO2 americium dioxide"),
    ("Cm(OH)3", "Cm(OH)3", "MIN", "Curium-hydroxide_1M_", "Cm(OH)3 curium hydroxide"),
    # ---- gases ------------------------------------------------------------------------------------------
    ("N2", "N2", "GA", "Nitrogen_1M_", "N2 nitrogen"),
    ("NH3(g)", "NH3", "GA", "Ammonia-gas_1M_", "NH3 ammonia gas"),
    ("CO", "CO", "GA", "Carbon-monoxide_1M_", "CO carbon monoxide"),
    ("SO2", "SO2", "GA", "Sulfur-dioxide_1M_", "SO2 sulfur dioxide"),
]


def build(icomp_file, base_file=None):
    icomp = {r["key"][0]: r for r in json.load(open(icomp_file, encoding="utf-8"))}
    existing = set()
    if base_file and os.path.exists(base_file):
        existing = {tuple(r["key"]) for r in json.load(open(base_file, encoding="utf-8"))}
        existing_names = {k[0] for k in existing}
    else:
        existing_names = set()
    template_keys = ("AUcl", "BC_dim", "BC_set", "BCname", "BCnote", "CA", "CD", "CDcl", "CI", "CIcl", "CmDCS",
                     "Ctext", "MasVol", "PCO", "bSDref", "bSDval", "formU", "symDC", "symIC", "unPCO")
    out, skipped = [], []
    for name, formula, kind, longname, descr in SPECIES:
        if name in existing_names:
            skipped.append(f"{name}: already in the base file")
            continue
        comp = dict(composition(formula))
        missing = [e for e in comp if e not in icomp]
        if missing:
            skipped.append(f"{name}: elements not in THEREDA {missing}")
            continue
        symbols = sorted(comp)
        amounts = [round(comp[s], 6) for s in symbols]
        amounts = [int(a) if float(a).is_integer() else a for a in amounts]
        mass = sum(a * icomp[s]["dod"]["M0i"] for s, a in zip(symbols, amounts)) / 1000.0
        dod = {k: None for k in template_keys}
        dod.update(
            BC_dim=[[len(symbols), 0, 0, 0, 0, 0]], BC_set=[["+", "-", "-", "-", "-", "-"]],
            BCname=f"{descr} 1 mol", BCnote="GEMS PCO database, extra salts for THEREDA elements",
            CI=[[a] for a in amounts], CIcl=[["M"] for _ in symbols], PCO=[[a] for a in amounts],
            symIC=[[f"{s:<6}{icomp[s]['key'][1]:<4}"] for s in symbols],
            MasVol=[[mass, 0, 0, 0, 0, 0, sum(amounts), 0]])
        out.append({"dod": dod, "key": [name, kind, longname.rstrip("_")], "project": "backup"})
    return out, skipped


if __name__ == "__main__":
    icomp_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data-out", "10_2026", "IComp.backup.json")
    base = os.path.join(HERE, "data-in", "Compos.backup.json")
    records, skipped = build(icomp_file, base)
    with open(os.path.join(HERE, "data-in", "Compos.extra.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=4, ensure_ascii=False)
    print(f"{len(records)} records written to data-in/Compos.extra.json")
    for line in skipped:
        print("skipped", line)
