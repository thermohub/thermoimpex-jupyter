import json
import re
import numpy as np
from collections import Counter

# -----------------------------------------------------------------------------
# JSON / aggregate-state utilities
# -----------------------------------------------------------------------------
def parse_json(obj):
    """Safely parse JSON from any object with .jsonString()."""
    try:
        return json.loads(obj.jsonString())
    except Exception:
        return {}

def is_state(substance, code: int, state_name: str) -> bool:
    """
    Generic check for aggregate_state.
    code: 0=gas,1=liquid,3=crystal,4=aqueous
    """
    data = parse_json(substance)
    return data.get("aggregate_state", {}).get(str(code)) == state_name

def is_gas(s):     return is_state(s, 0, "AS_GAS")
def is_liquid(s):  return is_state(s, 1, "AS_LIQUID")
def is_solid(s):   return is_state(s, 3, "AS_CRYSTAL")
def is_aqueous(s): return is_state(s, 4, "AS_AQUEOUS")

def contains_state(species_dict, database, code, state_name, skip_symbol=None):
    """
    Return True if any species in species_dict matches the given aggregate state.

    species_dict : dict of {symbol: coeff}
    database     : database object
    code         : aggregate_state code (0=gas,1=liquid,3=crystal,4=aqueous)
    state_name   : string, e.g. "AS_GAS", "AS_LIQUID", "AS_CRYSTAL", "AS_AQUEOUS"
    skip_symbol  : optional symbol to skip (e.g. the product species)
    """
    for key in species_dict:
        if skip_symbol and key == skip_symbol:
            continue
        substance = database.getSubstance(key)
        if substance and is_state(substance, code, state_name):
            return True
    return False


# --- Wrappers for readability ---

def contains_solid(species_dict, database, skip_symbol=None):
    return contains_state(species_dict, database, 3, "AS_CRYSTAL", skip_symbol)

def contains_gas(species_dict, database, skip_symbol=None):
    return contains_state(species_dict, database, 0, "AS_GAS", skip_symbol)

def contains_liquid(species_dict, database, skip_symbol=None):
    return contains_state(species_dict, database, 1, "AS_LIQUID", skip_symbol)

def contains_aqueous(species_dict, database, skip_symbol=None):
    return contains_state(species_dict, database, 4, "AS_AQUEOUS", skip_symbol)


# -----------------------------------------------------------------------------
# Generic property-presence checker
# -----------------------------------------------------------------------------
def all_have_property(r_dict: dict, database, prop_key: str) -> bool:
    """
    Return True if every symbol in r_dict exists in database
    and its JSON has prop_key.
    """
    for sym in r_dict:
        sub = database.getSubstance(sym)
        if sub is None:
            return False
        if prop_key not in parse_json(sub):
            return False
    return True

# Specialized wrappers
def all_have_sm_enthalpy(r_dict, database):
    return all_have_property(r_dict, database, "sm_enthalpy")

def all_have_sm_heat_capacity_p(r_dict, database):
    return all_have_property(r_dict, database, "sm_heat_capacity_p")

# -----------------------------------------------------------------------------
# Formula / string utilities
# -----------------------------------------------------------------------------
#def fix_formula(formula: str, rm_phase_sufix=None) -> str:
#    rm_phase_sufix = rm_phase_sufix or ['(l)', '(aq)', '(cr)', '(s)', '(orth)']
#    for p in rm_phase_sufix:
##        formula = formula.replace(p, "")
#    return formula.strip()

#def remove_substrings(s: str, sufixes_to_remove: list) -> str:
#    for sub in sufixes_to_remove:
 #       s = s.replace(sub, "")
#    return s

def remove_phase_sufix(s: str, sufixes_to_remove=None) -> str:
    sufixes_to_remove = sufixes_to_remove or ['(l)', '(aq)', '(cr)', '(s)', '(orth)']
    for sub in sufixes_to_remove:
        s = s.replace(sub, "")
    return s.strip()

def strip_formula_annotations(formula: str) -> str:
    # remove |...| blocks
    return re.sub(r'\|[^|]*\|', '', formula)

def expand_formula(formula: str) -> str:
    """
    Expand brackets like [ (UO2)2(AsO4) ]3 → (UO2)6(AsO4)3
    """
    pattern = re.compile(r'\[([^\[\]]+)\](\d+)')
    while True:
        m = pattern.search(formula)
        if not m:
            break
        inner, mult = m.groups()
        mult = int(mult)
        pieces = re.findall(r'(\([^()]+\))(\d*)', inner)
        expanded = "".join(f"{grp}{int(cnt or 1)*mult}" for grp, cnt in pieces)
        formula = formula[:m.start()] + expanded + formula[m.end():]
    return formula

def sum_elements(formula: str, targets: list) -> str:
    """
    Sum up counts of target elements and collapse them into first occurrence.
    """
    token = r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)"
    counts = {}
    positions = {}
    for m in re.finditer(token, formula):
        el, num = m.groups()
        if el in targets:
            val = float(num) if num else 1.0
            counts[el] = counts.get(el, 0.0) + val
            positions.setdefault(el, m.start())

    if not counts:
        return formula

    def fmt(el, v):
        v = round(v, 3)
        return el if v == 1 else f"{el}{v:g}"

    out = []
    i = 0
    while i < len(formula):
        m = re.match(token, formula[i:])
        if m:
            el, _ = m.groups()
            length = len(m.group(0))
            if el in counts and i == positions[el]:
                out.append(fmt(el, counts[el]))
            elif el not in counts:
                out.append(m.group(0))
            i += length
        else:
            out.append(formula[i])
            i += 1
    return "".join(out)

def clean_zero(val, threshold=1e-9):
    return 0.0 if abs(val) < threshold else val

# -----------------------------------------------------------------------------
# Reaction building
# -----------------------------------------------------------------------------
import re

def _build_reaction(symbol, formula, r_dict, 
                    mode="formation", 
                    prioritize=False, 
                    custom_reactant_func=None, 
                    fix_kwargs=None):
    """
    Generic PHREEQC reaction builder.

    Args:
        symbol (str): Species symbol.
        formula (str): Formula string.
        r_dict (dict): Reaction coefficients {species: coeff}.
        mode (str): "formation" or "dissolution" (affects sign).
        prioritize (bool): Whether to prioritize reactants with first element of symbol.
        custom_reactant_func (callable): Function to replace formula reactant with custom string.
        fix_kwargs (dict): Extra kwargs for fix_formula.
    """
    if fix_kwargs is None:
        fix_kwargs = {}

    # --- Step 1: coefficient and sign ---
    formula_coeff = r_dict.get(symbol, 0)
    if formula_coeff == 0:
        raise ValueError(f"Formula '{symbol}' not found in reaction dictionary.")

    sign = formula_coeff / abs(formula_coeff)

    if mode == "formation":
        sign = 1 if sign > 0 else -1
    elif mode == "dissolution":
        sign = 1 if sign < 0 else -1
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # --- Step 2: determine first element (for prioritize) ---
    first_element = ""
    if prioritize:
        match = re.match(r'([A-Z][a-z]?)', symbol)
        first_element = match.group(1) if match else ""

    reactants = []
    products = []

    for species, coeff in r_dict.items():
        if species == symbol:
            continue
        adjusted_coeff = coeff * sign
        abs_coeff = abs(adjusted_coeff)
        term = f"{abs_coeff:g} {remove_phase_sufix(species)}" if abs_coeff != 1 else remove_phase_sufix(species)
        if adjusted_coeff < 0:
            if prioritize:
                reactants.append((remove_phase_sufix(species), term))
            else:
                reactants.append(term)
        elif adjusted_coeff > 0:
            products.append(term)

    # --- Step 3: handle custom main reactant (dissolution vs. normal) ---
    if custom_reactant_func:
        main_reactant = custom_reactant_func(formula)
        reactants.insert(0, main_reactant)
    else:
        # Normal case: symbol is product, reactants from dict
        pass

    # --- Step 4: optional prioritization ---
    if prioritize and reactants:
        def element_priority(item):
            species, _ = item
            return (first_element not in species, species)
        reactants = [term for _, term in sorted(reactants, key=element_priority, reverse=True)]

    # --- Step 5: build reaction string ---
    if custom_reactant_func:
        reaction = " + ".join(reactants) + " = " + " + ".join(products)
    else:
        reaction = " + ".join(reactants) + f" = {remove_phase_sufix(symbol)}"
        if products:
            reaction += " + " + " + ".join(products)

    return reaction, sign


# === Wrappers for clarity ===

def to_phreeqc_formation_reaction(symbol, formula, r_dict):
    return _build_reaction(symbol, formula, r_dict, mode="formation")

def to_phreeqc_formation_reaction_prioritized(symbol, formula, r_dict):
    return _build_reaction(symbol, formula, r_dict, mode="formation", prioritize=True)

def to_phreeqc_dissolution_reaction(symbol, formula, r_dict):
    return _build_reaction(
        symbol, formula, r_dict, 
        mode="dissolution",
        custom_reactant_func=lambda f: sum_elements(expand_formula(strip_formula_annotations(f)), ['Fe','S']),
        fix_kwargs=None
    )

# -----------------------------------------------------------------------------
# Reaction property calculators
# -----------------------------------------------------------------------------
R = 8.31451
def calculate_A_coeffs(dHr, dGr, Cpr, T=298.15):
    """
    Returns dict A1..A6 for logK(T) = A1 + A2 T + A3/T + A4 log10(T) + A5 T^2 + A6/T^2
    """
    ln10 = np.log(10)
    A1 = ((dHr - dGr)/T - Cpr*(1 + np.log(T))) / (R * ln10)
    A2 = 0.0
    A3 = (Cpr*T - dHr) / (R * ln10)
    A4 = Cpr / R
    return {"A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": 0.0, "A6": 0.0}

def calculate_logK(T, **A):
    return (A["A1"] +
            A["A2"]*T +
            A["A3"]/T +
            A["A4"]*np.log10(T) +
            A["A5"]*T**2 +
            A["A6"]/T**2)

def is_logK_close(calc, exp, rtol=1e-4):
    return np.isclose(calc, exp, rtol=rtol)

## maybe I can give a json and type subst or reaction / substance used for check in generated and reaction in case of database record

def reaction_properties(engine, reaction=None, substance=None, react_dict=None, react_eq=None, database=None):

    if reaction:
        props_obj = reaction.thermoReferenceProperties()
        # 1) parse the reaction JSON once
        datar = parse_json(reaction)

        # 2) inspect JSON for enthalpy & heat-capacity keys
        has_dHr  = 'drsm_enthalpy' in datar 
        has_dCpr = 'drsm_heat_capacity_p' in datar 
        #print(has_dHr)
 
    elif substance and react_dict:
        props_obj = engine.thermoPropertiesReaction(298.15, 1e5, react_eq)
        datas = parse_json(substance)

        # 2) inspect JSON for enthalpy & heat-capacity keys
        has_dHr  = 'sm_enthalpy' in datas and all_have_sm_enthalpy(react_dict, database)
        has_dCpr = 'sm_heat_capacity_p' in datas and all_have_sm_heat_capacity_p(react_dict, database)
        #print(has_dHr)
    else:
        return {}

    # 3) pull the “always present” values
    T = 298.15
    Rln10 = 8.31451 * np.log(10)
    logKr = props_obj.log_equilibrium_constant.val
    Gr    = props_obj.reaction_gibbs_energy.val

    result = {
        "logK":  logKr,
        "elogK": props_obj.log_equilibrium_constant.err,
        "dGr":   Gr
    }

    # 4) only assign dHr / edHr if the JSON had it
    if has_dHr:
        Hr = props_obj.reaction_enthalpy.val
        result["dHr"]  = Hr
        result["edHr"] = props_obj.reaction_enthalpy.err
    else:
        result["dHr"]  = None
        result["edHr"] = None

    # 5) only assign dCpr / edCpr if the JSON had it
    if has_dCpr:
        Cpr = props_obj.reaction_heat_capacity_cp.val
        result["dCpr"]  = Cpr
        result["edCpr"] = props_obj.reaction_heat_capacity_cp.err
    else:
        Cpr = 0
        result["dCpr"]  = 0
        result["edCpr"] = 0

    # 6) compute analytical A1..A6 only when dHr is present
    if has_dHr: #and has_dCpr:
        A1 = ((Hr - Gr)/T - Cpr*(1 + np.log(T))) / Rln10
        A2 = 0.0
        A3 = (Cpr*T - Hr) / Rln10
        A4 = Cpr / 8.31451
        A5 = 0.0
        A6 = 0.0

        result.update({k: v for k, v in zip(
            ("A1","A2","A3","A4","A5","A6"),
             (A1,   A2,   A3,   A4,   A5,   A6)
        )})

        # optional consistency check
        calc = calculate_logK(T, A1=A1,A2=A2,A3=A3,A4=A4,A5=A5,A6=A6)
        if not is_logK_close(calc, logKr):
            print(f"{logKr} ❌ does not match computed {calc}")

    else:
        # explicitly mark them missing
        for k in ("A1","A2","A3","A4","A5","A6"):
            result[k] = None

    return result

def reaction_properties_json(propsJ: str):
    props = json.loads(propsJ)
    # pull out values & units, convert kJ→J if needed…
    # then call calculate_A_coeffs exactly as above…
    # (left as exercise)
    return {}

# -----------------------------------------------------------------------------
# PHREEQC formatter
# -----------------------------------------------------------------------------
class PhreeqcFormatter:
    def __init__(self, rm_phase_sufix=None):
        self.rm_phase_sufix = rm_phase_sufix or ['(l)', '(aq)', '(cr)', '(s)', '(orth)']
    
    def _has_analytical_data(self, props: dict) -> bool:
        return props.get("dHr") is not None #and props.get("dCpr") is not None

    def _fmt_volume(self, datas):
        lines = []
        vol = datas.get("sm_volume", {})
        if vol.get("units"):
            units = vol["units"][0]
            val, err = vol["values"][0], vol.get("errors", [None])[0]
            if units.startswith("J/bar"):
                val *= 10
                err = err*10 if err else None
                units = "cm^3/mol"
            l = f"\t-Vm\t{val:.3f}"
            l += f"\t# +- {err:.3f} {units}" if err else f"\t# {units}"
            lines.append(l)
        return lines

    def _fmt_enthalpy(self, datas=None, datar=None, sign=1, props=None, reactions_dict=None, database=None):
        lines = []

        if datar and 'drsm_enthalpy' in datar and datar['drsm_enthalpy'].get("units"):
            unit = datar['drsm_enthalpy']['units'][0]
        elif datas and 'sm_enthalpy' in datas and all_have_sm_enthalpy(reactions_dict, database) and datas['sm_enthalpy'].get("units"):
            unit = 'kJ/mol' #datas['sm_enthalpy']['units'][0]
        else:
            return lines  # No valid enthalpy data
        
        #print(props['dHr'] )
        l = f"\t# dHr {clean_zero(sign * props['dHr'] / 1000):.3f}"
        if props.get("edHr"):
            l += f" +- {clean_zero(props['edHr'] / 1000):.3f}"
        lines.append(l + f" {unit}")
        return lines

    def _fmt_heat_capacity(self, datas=None, datar=None, sign=1, props=None, reactions_dict=None, database=None):
        lines = []
        if datar and 'drsm_heat_capacity_p' in datar and datar['drsm_heat_capacity_p'].get("units"):
            unit = datar['drsm_heat_capacity_p']['units'][0]
        elif datas and 'sm_heat_capacity_p' in datas and all_have_sm_heat_capacity_p(reactions_dict, database) and datas['sm_heat_capacity_p'].get("units"):
            unit = datas['sm_heat_capacity_p']['units'][0]
        else:
            return lines  # No valid cp data
            
        l = f"\t# dCpr {clean_zero(sign*props['dCpr']):.3f}"
        if props.get("edCpr"):
            l += f" +- {clean_zero(props['edCpr']):.3f}"
        lines.append(l + f" {unit}")
        return lines

    def _fmt_analytical(self, sign, props):
        coefs = [props[k] * sign for k in ("A1","A2","A3","A4","A5","A6")]
        expr = " ".join(f"{clean_zero(c):.6f}" for c in coefs)
        return f"\t-analytical_expression {expr}"

    def _fmt_logK(self, sign, props):
        l = f"\t-log_K\t{sign*props['logK']:.4f}"
        if props.get("elogK"):
            l += f"\t# +- {props['elogK']:.4f}"
        return l

    def format_phase_reaction(self, reaction, substance, reaction_eq, sign, props):
        data_r = parse_json(reaction)
        data_s = parse_json(substance)
        name = substance.name().replace('_rdc_', '')
        lines = [name, "\t" + remove_phase_sufix(reaction_eq, self.rm_phase_sufix)]
        lines += self._fmt_volume(data_s)
        lines += self._fmt_enthalpy(None, data_r, sign, props)
        lines += self._fmt_heat_capacity(None, data_r, sign, props)
        if self._has_analytical_data(props):
            lines.append(self._fmt_analytical(sign, props))

        lines.append(self._fmt_logK(sign, props) + "\n")
        return "\n".join(lines)

    def format_phase_reaction_generated(self, substance, database, reactions_dict, reaction_eq, sign, props):
        #data_r = parse_json(reaction)
        datas = parse_json(substance)
        name = substance.name().replace('_rdc_', '')
        lines = [name, "\t" + remove_phase_sufix(reaction_eq, self.rm_phase_sufix)]
        lines += self._fmt_volume(datas)
        lines += self._fmt_enthalpy(datas, None, sign, props, reactions_dict, database)
        lines += self._fmt_heat_capacity(datas, None, sign, props, reactions_dict, database)
        if self._has_analytical_data(props):
            lines.append(self._fmt_analytical(sign, props))

        lines.append(self._fmt_logK(sign, props) + "\n")
        return "\n".join(lines)

    def format_aqueous_reaction(self, reaction, reaction_eq, sign, props):
        data = parse_json(reaction)
        lines = [remove_phase_sufix(reaction_eq, self.rm_phase_sufix)]
        lines += self._fmt_enthalpy(None, data, sign, props)
        lines += self._fmt_heat_capacity(None, data, sign, props)
        if self._has_analytical_data(props):
            lines.append(self._fmt_analytical(sign, props))
        lines.append(self._fmt_logK(sign, props) + "\n")
        return "\n".join(lines)

    def format_aqueous_reaction_generated(self, substance, database, reactions_dict, reaction_eq, sign, props):
        datas = json.loads(substance.jsonString())
        lines = [remove_phase_sufix(reaction_eq, self.rm_phase_sufix)]
        lines += self._fmt_enthalpy(datas, None, sign, props, reactions_dict, database)
        lines += self._fmt_heat_capacity(datas, None, sign, props, reactions_dict, database)
        if self._has_analytical_data(props):
            lines.append(self._fmt_analytical(sign, props))

        lines.append(self._fmt_logK(sign, props) + "\n")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Reactions process
# -----------------------------------------------------------------------------
import chemicalfun as cfun
def process_reaction(symbol, formula, formulas, symbols,
                     engine, database,
                     formatter_func,
                     mode="formation",
                     use_valence=False, retry_fallback=None):
    """
    Generic helper to generate a reaction, compute properties, and format output.

    Parameters
    ----------
    symbol : str
        Target species symbol.
    formula : str
        Formula string.
    formulas : list[str]
        Input formulas for ChemicalReactions.
    symbols : list[str]
        Input symbols for ChemicalReactions.
    engine : ThermoEngine
        Thermo engine object.
    database : Database
        Thermo database.
    formatter_func : callable
        Function(substance, database, r_dict, reac_eq, sign, props) -> str
    mode : str, default="formation"
        "formation" or "dissolution".
    use_valence : bool, default=False
        Whether to pass valence=True to ChemicalReactions.
    retry_fallback : tuple(formulas, symbols, use_valence), optional
        Fallback if ChemicalReactions raises RuntimeError.
    """

    # --- Step 1: Generate reaction dictionaries ---
    try:
        chem_rxns = cfun.ChemicalReactions(formulas, symbols, valence=use_valence)
        reactions = chem_rxns.generateReactions(formation=True)
    except RuntimeError:
        if retry_fallback:
            f2, s2, v2 = retry_fallback
            chem_rxns = cfun.ChemicalReactions(f2, s2, valence=v2)
            reactions = chem_rxns.generateReactions(formation=True)
        else:
            raise

    reactions_dic = [{el[0]: el[1] for el in r} for r in reactions]
    reactions_list = chem_rxns.stringReactions()

    # --- Step 2: Build PHREEQC reaction string ---
    if mode == "formation":
        reac_eq, sign = to_phreeqc_formation_reaction_prioritized(symbol, formula, reactions_dic[0])
    elif mode == "dissolution":
        reac_eq, sign = to_phreeqc_dissolution_reaction(symbol, formula, reactions_dic[0])
    else:
        raise ValueError(f"Invalid mode '{mode}': must be 'formation' or 'dissolution'")

    # --- Step 3: Compute thermo properties ---
    # Compute properties
    props = reaction_properties(engine, None, database.getSubstance(symbol),
                                 reactions_dic[0], reactions_list[0], database)

    # --- Step 4: Format output ---
    substance = database.getSubstance(symbol)
    return formatter_func(substance, database, reactions_dic[0], reac_eq, sign, props)



def correct_reactant_product_order(entries, product_symbol, output, reactants):
    """
    Insert a reaction entry into the list, making sure products appear
    after the reactions defining their reactants.
    
    entries : list of dicts [{symbol: {"out": str, "reactants": list}}, ...]
    product_symbol : str, product of this reaction
    output : str, formatted PHREEQC block
    reactants : list of reactant symbols (excluding product)
    """
    entry = {product_symbol: {"out": output, "reactants": reactants}}

    # default: append at the end
    insert_index = len(entries)
    for i, existing in enumerate(entries):
        existing_key = list(existing.keys())[0]
        existing_reactants = existing[existing_key]["reactants"]
        if product_symbol in existing_reactants:
            insert_index = i
            break

    entries.insert(insert_index, entry)



master_symbols = [
    'Ac+3', 'Ag+', 'Al+3', 'Am+3', 'HAsO4-2', 'B(OH)3(aq)', 'Ba+2', 'Br-', 'HCO3-', 'Ca+2', 'Cd+2', 'Cf+3',
    'Cit-3', 'Cl-', 'Cm+3', 'Cs+', 'Cu+2', 'e-', 'Edta-4', 'Eu+3', 'F-', 'Fe+2',
    'H2O(l)', 'H3Isa-',   'Hg+2', 'Ho+3', 'H+', 'HPO4-2', 'I-', 'K+',
    'Li+', 'Mg+2', 'Mn+2', 'MoO4-2', 'Na+', 'Nb(OH)4+', 'Ni+2', 'NO3-', 'NpO2+2',
    'Oxa-2', 'Pa+4', 'Pb+2', 'Pd+2', 'Po+2', 'PuO2+2', 'Ra+2', 'SeO3-2', 'Si(OH)4(aq)',
    'Sm+3', 'Sn+4', 'SO4-2', 'Sr+2', 'TcO(OH)2(aq)', 'Th+4', 'TiO+2', 'UO2+2',
    'Zn+2', 'Zr+4'
]

master_symbolsx = [
    'Ac+3', 'Ag+', 'Al+3', 'Am+3', 'B(OH)3(aq)', 'Ba+2', 'Br-', 'Ca+2', 'Cd+2', 'Cf+3',
    'Cit-3', 'Cl-', 'Cm+3', 'Cs+', 'Cu+2', 'Edta-4', 'Eu+3', 'F-', 'Fe+2',
    'H2O(l)', 'H3Isa-', 'HAsO4-2', 'HCO3-', 'Hg+2', 'Ho+3', 'H+', 'HPO4-2', 'I-', 'K+',
    'Li+', 'Mg+2', 'Mn+2', 'MoO4-2', 'Na+', 'Nb(OH)4+', 'Ni+2', 'NO3-', 'NpO2+2',
    'Oxa-2', 'Pa+4', 'Pb+2', 'Pd+2', 'Po+2', 'PuO2+2', 'Ra+2', 'SeO3-2', 'Si(OH)4(aq)',
    'Sm+3', 'Sn+4', 'SO4-2', 'Sr+2', 'TcO(OH)2(aq)', 'Th+4', 'TiO+2', 'UO2+2',
    'Zn+2', 'Zr+4'
]

# had to move HAs... and HCO3 in front
master_formulas = [
    'Ac+3', 'Ag+', 'Al+3', 'Am+3', 'HAs|+5|O4-2', 'B(OH)3@', 'Ba+2', 'Br-', 'HCO3-', 'Ca+2', 'Cd+2', 'Cf+3',
    'Cit-3', 'Cl-', 'Cm+3', 'Cs+', 'Cu|+2|+2', '-', 'Edta-4', 'Eu+3', 'F-', 'Fe+2',
    'H2O@', 'H3Isa-',   'Hg+2', 'Ho+3', 'H+', 'HPO4-2', 'I-', 'K+',
    'Li+', 'Mg+2', 'Mn+2', 'MoO4-2', 'Na+', 'Nb(OH)4+', 'Ni+2', 'NO3-', 'Np|+6|O2+2', 'Oxa-2',
    'Pa|+4|+4', 'Pb+2', 'Pd+2', 'Po|2|+2', 'Pu|+6|O2+2', 'Ra+2', 'Se|+4|O3-2', 'Si(OH)4@', 'Sm+3', 'Sn|+4|+4',
    'SO4-2', 'Sr+2', 'Tc|+4|O(OH)2@', 'Th+4', 'TiO+2', 'U|+6|O2+2', 'Zn+2', 'Zr+4'
]

master_formulasx = [
    'Ac+3', 'Ag+', 'Al+3', 'Am+3', 'B(OH)3@', 'Ba+2', 'Br-', 'Ca+2', 'Cd+2', 'Cf+3',
    'Cit-3', 'Cl-', 'Cm+3', 'Cs+', 'Cu|+2|+2', 'Edta-4', 'Eu+3', 'F-', 'Fe+2',
    'H2O@', 'H3Isa-', 'HAsO4-2', 'HCO3-', 'Hg+2', 'Ho+3', 'H+', 'HPO4-2', 'I-', 'K+',
    'Li+', 'Mg+2', 'Mn+2', 'MoO4-2', 'Na+', 'Nb(OH)4+', 'Ni+2', 'NO3-', 'Np|+6|O2+2', 'Oxa-2',
    'Pa|+4|+4', 'Pb+2', 'Pd+2', 'Po|2|+2', 'Pu|+6|O2+2', 'Ra+2', 'Se|+4|O3-2', 'Si(OH)4@', 'Sm+3', 'Sn|+4|+4',
    'SO4-2', 'Sr+2', 'Tc|+4|O(OH)2@', 'Th+4', 'TiO+2', 'U|+6|O2+2', 'Zn+2', 'Zr+4'
]

secondary_master = [
    'Ag(aq)', 'AmO2+', 'As(OH)3(aq)', 'B(OH)4-', 'CH4(aq)', 'CO3-2', 'ClO4-', 'Cu+', 'Fe+3', 'H2(aq)',
    'H2Se(aq)', 'HCN(aq)', 'Hg(aq)', 'HS-', 'HSeO4-', 'I2(aq)', 'IO3-', 'Mn+3', 'N2(aq)', 'NH4+',
    'Np+3', 'Np+4', 'NpO2+', 'O2(aq)', 'OH-', 'PO4-3', 'PaO(OH)+2', 'Pb(aq)', 'Pd(aq)', 'Po-2', 'Po(aq)',
    'Po+4', 'Pu+3', 'Pu+4', 'PuO2+', 'S(aq)', 'S2O3-2', 'Se(aq)', 'Sn+2', 'SO3-2',
    'TcO4-', 'Tc+3', 'Ti+3', 'U+4', 'UO2+'
]

secondary_master_formulas = [
    'Ag|0|@', 'Am|+5|O2+', 'As|+3|(OH)3@', 'B(OH)4-', 'C|-4|H4@', 'CO3-2', 'Cl|+7|O4-',
    'Cu|+1|+', 'Fe|+3|+3', 'H|0|2@', 'H2Se|-2|@', 'HC|0|N|-1|@', 'Hg|0|@', 'HS|-2|-',
    'HSe|+6|O4-', 'I|0|2@', 'I|+5|O3-', 'Mn|+3|+3', 'N|0|2@', 'N|-3|H4+', 'Np|+3|+3',
    'Np|+4|+4', 'Np|+5|O2+', 'O|0|2@', 'OH-', 'PO4-3', 'PaO(OH)+2', 'Pb|0|@', 'Pd|0|@',
    'Po|-2|-2', 'Po|0|@', 'Po+4', 'Pu|+3|+3', 'Pu|+4|+4', 'Pu|+5|O2+', 'S|0|',
    'S|+2|2O3-2', 'Se|0|', 'Sn+2', 'S|+4|O3-2', 'Tc|+7|O4-', 'Tc|+3|+3', 'Ti|+3|+3',
    'U|+4|+4', 'U|+5|O2+'
]

secondary_masterx = [
    'Ag(aq)', 'AmO2+', 'As(OH)3(aq)', 'CH4(aq)', 'ClO4-', 'Cu+', 'Fe+3', 'H2(aq)',
    'H2Se(aq)', 'HCN(aq)', 'Hg(aq)', 'HS-', 'HSeO4-', 'I2(aq)', 'IO3-', 'Mn+3', 'N2(aq)', 'NH4+',
    'Np+3', 'Np+4', 'NpO2+', 'O2(aq)',  'PaO(OH)+2', 'Pb(aq)', 'Pd(aq)', 'Po-2', 'Po(aq)',
    'Po+4', 'Pu+3', 'Pu+4', 'PuO2+', 'S(aq)', 'S2O3-2', 'Se(aq)', 'Sn+2', 'SO3-2',
    'TcO4-', 'Tc+3', 'Ti+3', 'U+4', 'UO2+'
]

secondary_master_formulasx = [
    'Ag|0|@', 'Am|+5|O2+', 'As|+3|(OH)3@', 'C|-4|H4@',  'Cl|+7|O4-',
    'Cu|+1|+', 'Fe|+3|+3', 'H|0|2@', 'H2Se|-2|@', 'HC|0|N|-1|@', 'Hg|0|@', 'HS|-2|-',
    'HSe|+6|O4-', 'I|0|2@', 'I|+5|O3-', 'Mn|+3|+3', 'N|0|2@', 'N|-3|H4+', 'Np|+3|+3',
    'Np|+4|+4', 'Np|+5|O2+', 'O|0|2@',  'PaO(OH)+2', 'Pb|0|@', 'Pd|0|@',
    'Po|-2|-2', 'Po|0|@', 'Po+4', 'Pu|+3|+3', 'Pu|+4|+4', 'Pu|+5|O2+', 'S|0|',
    'S|+2|2O3-2', 'Se|0|', 'Sn+2', 'S|+4|O3-2', 'Tc|+7|O4-', 'Tc|+3|+3', 'Ti|+3|+3',
    'U|+4|+4', 'U|+5|O2+'
]

# defined as DComp in tables
# other_aqueous = ['H2Po(aq)', 'Hg2+2', 'HPo-',  'PoCl4-2', 'PoCl6-2', 'PoSO4(aq)']

product_aqueous = [
    '(NpO2)2(OH)2+2', '(NpO2)2CO3(OH)3-', '(NpO2)3(CO3)6-6', '(NpO2)3(OH)5+', '(PuO2)2(OH)2+2',
    '(PuO2)3(CO3)6-6', '(UEdtaOH)2-2', '(UO2)2(OH)2+2', '(UO2)2CO3(OH)3-', '(UO2)2Edta(aq)',
    '(UO2)2NpO2(CO3)6-6', '(UO2)2OH+3', '(UO2)2PuO2(CO3)6-6', '(UO2)3(CO3)6-6', '(UO2)3(OH)4+2',
    '(UO2)3(OH)5+', '(UO2)3(OH)7-', '(UO2)3O(OH)2(HCO3)+', '(UO2)4(OH)7+', '(UO2Cit)2-2',
    'Ac(Cit)(aq)', 'Ac(Edta)-', 'Ac(OH)3(aq)', 'Ac(Oxa)+', 'Ac(Oxa)2-', 'AcCl+2', 'AcF+2',
    'AcF2+', 'AcF3(aq)', 'AcH2PO4+2', 'AcHEdta(aq)', 'AcOH+2', 'AcSCN+2', 'AcSO4+', 'Ag(CN)2-',
    'Ag(CN)3-2', 'Ag(HS)2-', 'Ag(OH)2-', 'Ag(OH)CN-', 'Ag(SeCN)3-2', 'Ag2S(HS)2-2',
    'Ag2Se(aq)', 'AgBr(aq)', 'AgBr2-', 'AgBr3-2', 'AgBr4-3', 'AgCl(aq)', 'AgCl2-', 'AgCl3-2',
    'AgCl4-3', 'AgCO3-', 'AgF(aq)', 'AgHPO4-', 'AgHS(aq)', 'AgI(aq)', 'AgI2-', 'AgI3-2',
    'AgI4-3', 'AgOH(aq)', 'AgSeO3-', 'AgSeO4-', 'AgSO4-', 'Al(OH)2+', 'Al(OH)2F(aq)',
    'Al(OH)2F2-', 'Al(OH)3(aq)', 'Al(OH)4-', 'Al(SO4)2-', 'Al13(OH)32+7', 'Al2(OH)2+4',
    'Al3(OH)4+5', 'AlF+2', 'AlF2+', 'AlF3(aq)', 'AlF4-', 'AlF5-2', 'AlF6-3', 'AlOH+2',
    'AlOHF2(aq)', 'AlSiO(OH)3+2', 'AlSO4+', 'Am(Cit)(aq)', 'Am(Cit)2-3', 'Am(CO3)2-', 'Am(CO3)3-3',
    'Am(Edta)-', 'Am(Edta)OH-2', 'Am(HCit)+', 'Am(HCit)2-', 'Am(HEdta)(aq)', 'Am(Isa)-',
    'Am(NO3)2+', 'Am(OH)2+', 'Am(OH)3(aq)', 'Am(Oxa)+', 'Am(Oxa)2-', 'Am(Oxa)3-3',
    'Am(SO4)2-', 'AmCl+2', 'AmCl2+', 'AmCO3+', 'AmF+2', 'AmF2+', 'AmH2PO4+2', 'AmHCO3+2',
    'AmHPO4+', 'AmNO3+2', 'AmO2(CO3)2-3', 'AmO2(CO3)3-5', 'AmO2(OH)2-', 'AmO2CO3-', 'AmO2OH(aq)',
    'AmOH+2', 'AmSCN+2', 'AmSiO(OH)3+2', 'AmSO4+', 'As(OH)4-', 'AsO4-3', 'Ba2UO2(CO3)3(aq)',
    'BaCO3(aq)', 'BaF+', 'BaH2PO4+', 'BaHCO3+', 'BaHPO4(aq)', 'BaOH+', 'BaPO4-', 'BaSO4(aq)',
    'BaUO2(CO3)3-2', 'Ca(Cit)-', 'Ca(Edta)-2', 'Ca(H2Cit)+', 'Ca(H2Isa)(aq)', 'Ca(H3Isa)+',
    'Ca(HCit)(aq)', 'Ca(HEdta)-', 'Ca(Oxa)(aq)', 'Ca(Oxa)2-2', 'Ca2Am(OH)4+3', 'Ca2Cm(OH)4+3',
    'Ca2UO2(CO3)3(aq)', 'Ca2Zr(OH)6+2', 'Ca3Am(OH)6+3', 'Ca3Cm(OH)6+3', 'Ca3NpO2(OH)5+2',
    'Ca3Zr(OH)6+4', 'Ca4Np(OH)8+4', 'Ca4Pu(OH)8+4', 'Ca4Th(OH)8+4', 'CaAm(OH)3+2',
    'CaCm(OH)3+2', 'CaCO3(aq)', 'CaF+', 'CaH2PO4+', 'CaHCO3+', 'CaHPO4(aq)', 'CaMoO4(aq)',
    'CaNpO2(OH)2+', 'CaOH+', 'CaPO4-', 'CaSeO4(aq)', 'CaSiO(OH)3+', 'CaSiO2(OH)2(aq)',
    'CaSO4(aq)', 'CaUO2(CO3)3-2', 'CaZr(OH)6(aq)', 'Cd(CO3)2-2', 'Cd(HS)2(aq)', 'Cd(HS)3-',
    'Cd(HS)4-2', 'Cd(OH)2(aq)', 'Cd(OH)3-', 'Cd(OH)4-2', 'Cd(SO4)2-2', 'Cd2OH+3', 'CdCl+',
    'CdCl2(aq)', 'CdCl3-', 'CdCl4-2', 'CdCO3(aq)', 'CdH2PO4+', 'CdHCO3+', 'CdHPO4(aq)',
    'CdHS+', 'CdOH+', 'CdS(HS)-', 'CdSO4(aq)', 'CfF+2', 'CfSCN+2', 'CfSO4+', 'Cm(CO3)2-',
    'Cm(CO3)3-3', 'Cm(NO3)2+', 'Cm(OH)2+', 'Cm(OH)3(aq)', 'Cm(SO4)2-', 'CmCl+2', 'CmCl2+',
    'CmCO3+', 'CmF+2', 'CmF2+', 'CmH2PO4+2', 'CmHCO3+2', 'CmHPO4+', 'CmNO3+2', 'CmOH+2',
    'CmSCN+2', 'CmSiO(OH)3+2', 'CmSO4+', 'CN-', 'CO2(aq)', 'Cu(CO3)2-2', 'Cu(CO3)OH-',
    'Cu(H2PO4)(HPO4)-', 'Cu(H2PO4)(HPO4)-2', 'Cu(H2PO4)2-', 'Cu(H2PO4)2(aq)', 'Cu(HPO4)2-2',
    'Cu(HS)2-', 'Cu(OH)2-', 'Cu(OH)2(aq)', 'Cu(OH)3-', 'Cu(OH)4-2', 'Cu2(OH)2+2', 'Cu2Cl4-2',
    'Cu2OH+3', 'Cu2S(HS)2-2', 'Cu3(OH)4+2', 'CuCl(aq)', 'CuCl+', 'CuCl2-', 'CuCl2(aq)',
    'CuCl3-2', 'CuCO3(aq)', 'CuH2PO4(aq)', 'CuH2PO4+', 'CuHCO3+', 'CuHPO4(aq)', 'CuHS(aq)',
    'CuOH(aq)', 'CuOH+', 'CuSO4(aq)', 'Eu(CO3)2-', 'Eu(OH)2+', 'Eu(OH)3(aq)', 'Eu(OH)4-',
    'Eu(SO4)2-', 'EuCO3+', 'EuF+2', 'EuF2+', 'EuH2PO4+2', 'EuOH+2', 'EuSiO(OH)3+2', 'EuSO4+',
    'Fe(CN)6-3', 'Fe(CN)6-4', 'Fe(CO3)2-2', 'Fe(CO3)3-3', 'Fe(H2PO4)3(aq)', 'Fe(HSeO4)2(aq)',
    'Fe(OH)2(aq)', 'Fe(OH)2+', 'Fe(OH)3-', 'Fe(OH)3(aq)', 'Fe(OH)4-', 'Fe(OH)CO3(aq)',
    'Fe(SCN)2+', 'Fe(SO4)2-', 'Fe(SO4)2-2', 'Fe2(OH)2+4', 'Fe3(OH)4+5', 'FeCl+', 'FeCl+2',
    'FeCl2+', 'FeCl3(aq)', 'FeCl4-', 'FeCO3(aq)', 'FeF+', 'FeF+2', 'FeF2+', 'FeH2PO4+',
    'FeHPO4(aq)', 'FeHSeO3(H2SeO3)+', 'FeHSeO3+', 'FeHSO4+', 'FeHSO4+2', 'FeOH+', 'FeOH+2',
    'FePO4(aq)', 'FeS(aq)', 'FeSCN+2', 'FeSeO3+', 'FeSiO(OH)3+2', 'FeSO4(aq)', 'FeSO4+',
    'H2AsO4-', 'H2Cit-', 'H2Edta-2', 'H2Fe(CN)6-2', 'H2MoO4(aq)', 'H2Nb6O19-6', 'H2Oxa(aq)',
     'H2Po(aq)',         'H2PO4-', 'H2S(aq)', 'H2SeO3(aq)', 'H3AsO4(aq)', 'H3Cit(aq)', 'H3Edta-', 'H3Nb6O19-5',
    'H3PO4(aq)', 'H4Edta(aq)', 'H4Isa(aq)', 'H5Edta+', 'H6Edta+2', 'HCit-2', 'HEdta-3', 'HF(aq)',
    'HF2-', 'HFe(CN)6-3', 'Hg(HS)2(aq)', 'Hg(OH)2(aq)', 'Hg(OH)3-', 'Hg(SO4)2-2',          'Hg2+2',
    'Hg2OH+', 'HgCl+', 'HgCl2(aq)', 'HgCl3-', 'HgCl4-2', 'HgCO3(aq)', 'HgHCO3+', 'HgHPO4(aq)',
    'HgOH+', 'HgOHCl(aq)', 'HgOHCO3-', 'HgPO4-', 'HgS(HS)-', 'HgS2-2', 'HgSO4(aq)', 'HIO3(aq)',
    'HMoO4-', 'HNb6O19-7', 'Ho(CO3)2-', 'Ho(OH)2+', 'Ho(OH)3(aq)', 'Ho(OH)4-', 'HoSiO(OH)3+2',
    'Ho(SO4)2-', 'HoCO3+', 'HoF+2', 'HoF2+', 'HoH2PO4+2', 'HoOH+2', 'HoSO4+', 'HOxa-',
    'HP2O7-3',     'HPo-', 'HSe-', 'HSe4-', 'HSeO3-', 'HSO3-', 'HSO4-', 'I3-', 'KCO3-', 'KEdta-3',
    'KFe(CN)6-2', 'KFe(CN)6-3', 'KH2PO4(aq)', 'KHCO3(aq)', 'KHPO4-', 'KOH(aq)', 'KPO4-2',
    'KSO4-', 'LiF(aq)', 'LiH2PO4(aq)', 'LiHPO4-', 'LiOH(aq)', 'LiPO4-2', 'LiSO4-', 'Mg(Cit)-',
    'Mg(Edta)-2', 'Mg(H2Cit)+', 'Mg(HCit)(aq)', 'Mg(HEdta)-', 'Mg(Oxa)(aq)', 'Mg(Oxa)2-2',
    'Mg2UO2(CO3)3(aq)', 'MgCO3(aq)', 'MgF+', 'MgH2PO4+', 'MgHCO3+', 'MgHPO4(aq)', 'MgOH+',
    'MgPO4-', 'MgSeO4(aq)', 'MgSiO(OH)3+', 'MgSiO2(OH)2(aq)', 'MgSO4(aq)', 'MgUO2(CO3)3-2',
    'Mn(OH)2(aq)', 'Mn(OH)2+', 'Mn(OH)3-', 'Mn(OH)4-2', 'MnCl+', 'MnCl+2', 'MnCO3(aq)',
    'MnF+', 'MnF+2', 'MnF2+', 'MnF3(aq)', 'MnHCO3+', 'MnOH+', 'MnOH+2', 'MnOHF+', 'MnSeO4(aq)',
    'MnSO4(aq)', 'Na(Edta)-3', 'NaCO3-', 'NaF(aq)', 'NaH2PO4(aq)', 'NaHCO3(aq)', 'NaHPO4-',
    'NaOH(aq)', 'NaPO4-2', 'NaSO4-', 'Nb(OH)5(aq)', 'Nb(OH)6-', 'Nb(OH)7-2', 'Nb6O19-8',
    'NH3(aq)', 'Ni(Cit)-', 'Ni(Cit)2-4', 'Ni(CN)4-2', 'Ni(CN)5-3', 'Ni(CO3)2-2', 'Ni(Edta)-2',
    'Ni(H2Cit)+', 'Ni(H3Isa)+', 'Ni(HCit)(aq)', 'Ni(HEdta)-', 'Ni(HS)2(aq)', 'Ni(NH3)2+2',
    'Ni(NH3)3+2', 'Ni(NH3)4+2', 'Ni(NH3)5+2', 'Ni(NH3)6+2', 'Ni(OH)2(aq)', 'Ni(OH)3-', 'Ni(Oxa)(aq)',
    'Ni(Oxa)2-2', 'Ni(SCN)2(aq)', 'Ni(SCN)3-', 'Ni(SeCN)2(aq)', 'Ni2OH+3', 'Ni4(OH)4+4',
    'NiCl+', 'NiCO3(aq)', 'NiF+', 'NiHAsO4(aq)', 'NiHCO3+', 'NiHP2O7-', 'NiHPO4(aq)', 'NiHS+',
    'NiNH3+2', 'NiNO3+', 'NiOH+', 'NiP2O7-2', 'NiSCN+', 'NiSeCN+', 'NiSeO4(aq)', 'NiSiO(OH)3+',
    'NiSO4(aq)', 'Np(CO3)2-', 'Np(CO3)3-3', 'Np(CO3)4-4', 'Np(CO3)5-6', 'Np(Edta)(aq)', 'Np(OH)2(CO3)2-2',
    'Np(OH)2+', 'Np(OH)2+2', 'Np(OH)3(aq)', 'Np(OH)3(H3Isa)(aq)', 'Np(OH)3(H3Isa)2-', 'Np(OH)3+',
    'Np(OH)4(aq)', 'Np(OH)4(H3Isa)-', 'Np(OH)4(H3Isa)2-2', 'Np(OH)4CO3-2', 'Np(Oxa)+2',
    'Np(Oxa)2(aq)', 'Np(Oxa)3-2', 'Np(SCN)2+2', 'Np(SCN)3+', 'Np(SO4)2-', 'Np(SO4)2(aq)',
    'NpCl+2', 'NpCl+3', 'NpCl2+', 'NpCO3+', 'NpF+2', 'NpF+3', 'NpF2+', 'NpF2+2', 'NpI+3',
    'NpNO3+3', 'NpO2(CO3)2-2', 'NpO2(CO3)2-3', 'NpO2(CO3)2OH-4', 'NpO2(CO3)3-4', 'NpO2(CO3)3-5',
    'NpO2(H2Edta)-', 'NpO2(HEdta)-2', 'NpO2(HPO4)2-2', 'NpO2(OH)(aq)', 'NpO2(OH)2-', 'NpO2(OH)2(aq)', 'NpO2(OH)3-', 'NpO2(OH)4-2',
    'NpO2(Oxa)2-3', 'NpO2(SO4)2-2', 'NpO2Cit-2', 'NpO2Cl+', 'NpO2CO3-', 'NpO2CO3(aq)', 'NpO2CO3(OH)2-3',
    'NpO2CO3OH-2', 'NpO2Edta-3', 'NpO2EdtaOH-4', 'NpO2F(aq)', 'NpO2F+', 'NpO2F2-', 'NpO2F2(aq)',
    'NpO2H2PO4(aq)', 'NpO2H2PO4+', 'NpO2HPO4-', 'NpO2HPO4(aq)', 'NpO2IO3(aq)', 'NpO2IO3+',
    'NpO2OH+', 'NpO2Oxa-', 'NpO2SCN(aq)', 'NpO2SiO(OH)3(aq)', 'NpO2SiO(OH)3+', 'NpO2SO4-',
    'NpO2SO4(aq)', 'NpOH+2', 'NpOH+3', 'NpSCN+3', 'NpSiO(OH)3+2', 'NpSiO(OH)3+3', 'NpSO4+',
    'NpSO4+2',      'P2O7-4', 'Pa(OH)2+2', 'Pa(OH)3+', 'PaO(OH)2+', 'PaO(OH)3(aq)',
    'PaO(OH)4-', 'PaO(SO4)2-', 'PaO(SO4)3-3', 'PaO+3', 'PaOF+2', 'PaOF2+', 'PaOF3(aq)',
    'PaOH+3', 'PaOSO4+', 'Pb(CO3)2-2', 'Pb(CO3)Cl-', 'Pb(CO3)OH-', 'Pb(HPO4)2-2', 'Pb(HS)2(aq)',
    'Pb(OH)2(aq)', 'Pb(OH)3-', 'Pb(SO4)2-2', 'Pb2OH+3', 'Pb3(OH)4+2', 'Pb4(OH)4+4',
    'Pb6(OH)8+4', 'PbCl+', 'PbCl2(aq)', 'PbCl3-', 'PbCl4-2', 'PbCO3(aq)', 'PbH2PO4+',
    'PbHCO3+', 'PbHPO4(aq)', 'PbOH+', 'PbS(HS)-', 'PbSO4(aq)', 'Pd(NH3)2+2', 'Pd(NH3)3+2',
    'Pd(NH3)4+2', 'Pd(OH)2(aq)', 'Pd(OH)3-', 'PdCl+', 'PdCl2(aq)', 'PdCl3-', 'PdCl4-2',
    'PdNH3+2', 'PdOH+', 'Po(NO3)2+2', 'Po(NO3)3+', 'Po(OH)2+2', 'Po(OH)3+', 'Po(OH)4(aq)',
    'Po(OH)6-2', 'Po(OH)Cl4-', 'Po(SO4)2(aq)', 'Po(SO4)3-2',      'PoCl+', 'PoCl2(aq)',
    'PoCl3-',            'PoCl4-2', 'PoCl6-2', 'PoNO3+3', 'PoOH+3',      'PoSO4(aq)', 'PoSO4+2', 'Pu(Cit)(aq)',
    'Pu(CO3)2-', 'Pu(CO3)3-3', 'Pu(CO3)4-4', 'Pu(CO3)5-6', 'Pu(Edta)-', 'Pu(Edta)(aq)', 'Pu(Edta)(OH)2-2',
    'Pu(Edta)OH-', 'Pu(HCit)+', 'Pu(HEdta)(aq)', 'Pu(OH)2+', 'Pu(OH)2+2', 'Pu(OH)3(aq)', 'Pu(OH)3+',
    'Pu(OH)4(aq)', 'Pu(Oxa)+', 'Pu(Oxa)+2', 'Pu(Oxa)2-', 'Pu(Oxa)2(aq)', 'Pu(Oxa)3-2',
    'Pu(Oxa)3-3', 'Pu(SO4)2-', 'Pu(SO4)2(aq)', 'PuCl+2', 'PuCl+3', 'PuCO3(OH)3-', 'PuCO3+',
    'PuF+2', 'PuF+3', 'PuF2+', 'PuF2+2', 'PuH2PO4+2', 'PuH3PO4+4', 'PuNO3+3', 'PuO2(CO3)2-2',
    'PuO2(CO3)2-3', 'PuO2(CO3)3-4', 'PuO2(CO3)3-5', 'PuO2(H2PO4)2(aq)', 'PuO2(OH)2(aq)',
    'PuO2(OH)3-', 'PuO2(SO4)2-2', 'PuO2Cl+', 'PuO2Cl2(aq)', 'PuO2CO3-', 'PuO2CO3(aq)', 'PuO2F+', 'PuO2F2(aq)',
    'PuO2H2PO4+', 'PuO2HPO4(aq)', 'PuO2OH(aq)', 'PuO2OH+', 'PuO2PO4-', 'PuO2SiO(OH)3(aq)',
    'PuO2SiO(OH)3+', 'PuO2SO4-', 'PuO2SO4(aq)', 'PuOH+2', 'PuOH+3', 'PuSCN+2', 'PuSiO(OH)3+2',
    'PuSiO(OH)3+3', 'PuSO4+','PuSO4+2', 'RaCO3(aq)', 'RaF+', 'RaHCO3+', 'RaOH+', 'RaSO4(aq)', 'S-2', 'SCN-', 'Se-2', 'Se2-2',
    'Se3-2', 'Se4-2', 'SeCN-', 'SeO4-2', 'Si4O8(OH)4-4', 'SiAlO3(OH)4-3', 'SiO(OH)3-', 'SiO2(OH)2-2',
    'Sm(CO3)2-', 'Sm(OH)2+', 'Sm(OH)3(aq)', 'Sm(OH)4-', 'Sm(SO4)2-', 'Sm2(OH)2+4', 'Sm3(OH)5+4',
    'SmCO3+', 'SmF+2', 'SmF2+', 'SmH2PO4+2', 'SmOH+2', 'SmSiO(OH)3+2', 'SmSO4+', 'Sn(H2PO4)2(aq)',
    'Sn(HPO4)2-2', 'Sn(HPO4)3-4', 'Sn(NO3)2(aq)', 'Sn(OH)2(aq)', 'Sn(OH)3-', 'Sn(OH)4(aq)',
    'Sn(OH)5-', 'Sn(OH)6-2', 'Sn(OH)Cl(aq)', 'Sn3(OH)4+2', 'SnBr+', 'SnBr2(aq)', 'SnBr3-',
    'SnCl+', 'SnCl+3', 'SnCl2(aq)', 'SnCl2+2', 'SnCl3-', 'SnCl4-2', 'SnCl4(aq)', 'SnCl5-',
    'SnCl6-2', 'SnF+', 'SnF2(aq)', 'SnF3-', 'SnF6-2', 'SnH2PO4+', 'SnH2PO4HPO4-', 'SnHPO4(aq)',
    'SnNO3+', 'SnOH+', 'SnPO4-', 'SnSCN+', 'SnSO4(aq)', 'Sr2UO2(CO3)3(aq)', 'SrCO3(aq)', 'SrF+',
    'SrH2PO4+', 'SrHCO3+', 'SrHPO4(aq)', 'SrOH+', 'SrPO4-', 'SrSO4(aq)', 'SrUO2(CO3)3-2',
    'Tc2O2(OH)2+2', 'Tc2OCl10-4', 'TcCl5-', 'TcCl6-2', 'TcCO3(OH)2(aq)', 'TcCO3(OH)3-', 'TcO(OH)3-',
    'Th(CO3)5-6', 'Th(H2PO4)2+2', 'Th(H3PO4)(H2PO4)+3', 'Th(IO3)2+2', 'Th(IO3)3+', 'Th(NO3)2+2',
    'Th(OH)2(CO3)2-2', 'Th(OH)2+2', 'Th(OH)2CO3(aq)', 'Th(OH)3(SiO(OH)3)3-2', 'Th(OH)3CO3-',
    'Th(OH)4(aq)', 'Th(OH)4CO3-2', 'Th(SCN)2+2', 'Th(SO4)2(aq)', 'Th(SO4)3-2', 'Th2(OH)2+6',
    'Th2(OH)3+5', 'Th4(OH)12+4', 'Th4(OH)8+8', 'Th6(OH)14+10', 'Th6(OH)15+9', 'ThCl+3',
    'ThF+3', 'ThF2+2', 'ThF3+', 'ThF4(aq)', 'ThF6-2', 'ThH2PO4+3', 'ThH3PO4+4', 'ThIO3+3',
    'ThNO3+3', 'ThOH(CO3)4-5', 'ThOH+3', 'ThSCN+3', 'ThSO4+2', 'Ti2(OH)2+4', 'TiO(OH)2(aq)',
    'TiO(OH)3-', 'TiOH+2', 'TiOOH+', 'U(CO3)4-4', 'U(CO3)5-6', 'U(NO3)2+2', 'U(OH)2+2',
    'U(OH)3+', 'U(OH)4(aq)', 'U(Oxa)2(aq)', 'U(Oxa)3-2', 'U(Oxa)4-4', 'U(SCN)2+2', 'U(SO4)2(aq)',
    'UCl+3', 'UCO3(OH)3-', 'UEdta(aq)', 'UEdta(OH)2-2', 'UEdtaOH-', 'UF+3', 'UF2+2', 'UF3+',
    'UF4(aq)', 'UF5-', 'UF6-2', 'UH2PO4+3', 'UI+3', 'UNO3+3', 'UO2(CO3)2-2', 'UO2(CO3)3-4',
    'UO2(CO3)3-5', 'UO2(H2AsO4)2(aq)', 'UO2(H2PO4)(H3PO4)+', 'UO2(H2PO4)2(aq)', 'UO2(H3Isa)+',
    'UO2(H3Isa)2(aq)', 'UO2(H3Isa)3-', 'UO2(HCit)(aq)', 'UO2(HEdta)-', 'UO2(HSeO3)2(aq)', 'UO2(IO3)2(aq)',
    'UO2(OH)2(aq)', 'UO2(OH)3-', 'UO2(OH)4-2', 'UO2(Oxa)2-2', 'UO2(Oxa)3-4', 'UO2(SCN)2(aq)',
    'UO2(SCN)3-', 'UO2(SeO4)2-2', 'UO2(SO4)2-2', 'UO2(SO4)3-4', 'UO2Cit-', 'UO2Cl+', 'UO2Cl2(aq)',
    'UO2CO3(aq)', 'UO2CO3F-', 'UO2CO3F2-2', 'UO2CO3F3-3', 'UO2Edta-2', 'UO2F+', 'UO2F2(aq)',
    'UO2F3-', 'UO2F4-2', 'UO2H2AsO4+', 'UO2H2PO4+', 'UO2H3PO4+2', 'UO2HAsO4(aq)', 'UO2HPO4(aq)',
    'UO2HSeO3+', 'UO2IO3+', 'UO2NO3+', 'UO2OH+', 'UO2Oxa(aq)', 'UO2PO4-', 'UO2SCN+', 'UO2SeO4(aq)',
    'UO2SiO(OH)3+', 'UO2SO4(aq)', 'UOH+3', 'UOxa+2', 'USCN+3', 'USO4+2', 'Zn(CO3)2-2',
    'Zn(H2PO4)(HPO4)-', 'Zn(H2PO4)2(aq)', 'Zn(HPO4)(PO4)-3', 'Zn(HPO4)2-2', 'Zn(HPO4)3-4',
    'Zn(HS)2(aq)', 'Zn(HS)3-', 'Zn(OH)2(aq)', 'Zn(OH)2HPO4-2', 'Zn(OH)3-', 'Zn(OH)4-2',
    'Zn(SO4)2-2', 'Zn2CO3+2', 'Zn2OH+3', 'ZnCl+', 'ZnCl2(aq)', 'ZnCl3-', 'ZnCl4-2', 'ZnCO3(aq)',
    'ZnH2PO4+', 'ZnHCO3+', 'ZnHPO4(aq)', 'ZnOH+', 'ZnS(HS)-', 'ZnSO4(aq)', 'Zr(CO3)4-4',
    'Zr(NO3)2+2', 'Zr(OH)2+2', 'Zr(OH)4(aq)', 'Zr(OH)6-2', 'Zr(SO4)2(aq)', 'Zr(SO4)3-2',
    'Zr3(OH)4+8', 'Zr3(OH)9+3', 'Zr4(OH)15+', 'Zr4(OH)16(aq)', 'Zr4(OH)8+8', 'ZrCl+3', 'ZrCl2+2', 'ZrF+3', 'ZrF2+2',
    'ZrF3+', 'ZrF4(aq)', 'ZrF5-', 'ZrF6-2', 'ZrNO3+3', 'ZrOH+3', 'ZrSO4+2'
]


product_solids = [
    'Mercury', '(HgOH)3PO4(s)', '(NH4)4NpO2(CO3)3(s)', '(PuO2)3(PO4)2:4(H2O)(am)',
    'Soddyite', 'Trögerite', '(UO2)3(PO4)2:4H2O(cr)', 'Ac(OH)3(aged)',
    'Ac(OH)3(fresh)', 'Ac2(Oxa)3(s)', 'Al(OH)3(s)', 'Al(OOH)(alpha)', 'Al2O3(alpha)', 'Silver', 'Ag2CO3(cr)',
    'Ag2O(am)', 'Ag2O(cr)', 'Ag2S(cr)', 'Ag2Se(alpha)',
    'Ag2SeO3(cr)', 'Ag2SeO4(cr)', 'Ag2SO4(s)', 'Ag3PO4(s)',
    'AgBr(cr)', 'AgCl(cr)', 'AgCN(s)', 'AgI(cr)',
    'AgSeCN(cr)', 'Gibbsite', 'Kaolinite', 'Boehmite',
    'Am(OH)3(am)', 'Am(OH)3(cr)', 'Am2(CO3)3(am_hyd)', 'AmO2OH(am)',
    'AmOHCO3(am_hyd)', 'AmOHCO3:0.5H2O(cr)', 'Ba[(UO2)(AsO4)]2:7H2O(cr)',
    'Meta-uranocircite_II', 'Ba3(PO4)2(s)', 'Witherite', 'BaHPO4(cr)',
    'BaSeO3(cr)', 'BaSeO4(cr)', 'Barite', 'Beidellite_SBld-1',
    'Beidellite(Ca)', 'Beidellite(K)', 'Beidellite(Mg)', 'Beidellite(Na)',
    'Berthierine_ISGS', 'Berthierine(FeII)', 'Berthierine(FeIII)', 'C3AS3(cr)', 'C3FS3(cr)', 'CaSO4w0.5(cr)',  'CaSiO3(cr)',  'Ca(H3Isa)2(cr)',
    'Portlandite', 'Whewellite', 'Weddelite', 'Ca(Oxa):3H2O(cr)',
    'Uranophane', 'Ca[(UO2)(AsO4)]2:10H2O(cr)', 'Meta-autunite', 'Ca0.5NpO2(OH)2:1.3H2O(cr)',
    'Ca3(Cit)2:4H2O(cr)', 'Tuite', 'Ca4H(PO4)3:2.5H2O(s)', 'Cl-apatite',
    'F-apatite', 'OH-apatite', 'Aragonite', 'Calcite',
    'Vaterite', 'Fluorite', 'CaHK3(PO4)2(cr)', 'Monetite',
    'Brushite', 'Dolomite', 'Powellite', 'Cancrinite-NO3',
    'CaNpO2(OH)2.6Cl0.4:2H2O(cr)', 'CaSeO3:H2O(cr)', 'CaSn(OH)6(pr)', 'Anhydrite',
    'Gypsum', 'Becquerelite', 'Cd(OH)2(s)', 'Cd[(UO2)(AsO4)]2:8H2O(cr)',
    'Cd[(UO2)(PO4)]2:10H2O(cr)', 'Cd3(PO4)2(s)', 'Cd5H2(PO4)4:4H2O(s)', 'Otavite',
    'CdS(s)', 'Chabazite-Ca', 'Chabazite-Na', 'Clinoptilolite',
    'Cronstedtite', 'Cs(UO2)(AsO4):2.5H2O(cr)', 'Cs(UO2)(BO3):H2O(cr)', 'Cs(UO2)(PO4):2.5H2O(cr)',
    'Cu(OH)2(s)', 'Metazeunerite', 'Cu[(UO2)(PO4)]2:8H2O(cr)', 'Malachite',
    'Cuprite', 'Chalcotite', 'Azurite', 'CuCl(s)',
    'Tenorite', 'Covellite', 'Eu(OH)3(am)', 'Eu(OH)3(cr)',
    'Eu2(CO3)3(cr)', 'EuF3(cr)', 'EuOHCO3(cr)', 'Eu-rhabdophane',
    'Faujasite-X', 'Faujasite-Y', 'Fe(OH)2(s)', 'Ferrihydrite-2line',
    'Metakahlerite', 'Pyrrhotite-4C', 'Fe0.875Se(cr)', 'Pyrrhotite-5C',
    'Fe1.042Se(cr)', 'Fe2(SeO3)3:3H2O(cr)', 'Fe-hibbingite', 'Hematite',
    'Maghemite', 'Vivianite', 'Magnetite', 'Greigite',
    'Fe3Se4(gamma)', 'Fe4(OH)8Cl:nH2O(s)', 'Fe6(OH)12CO3:nH2O(s)', 'Fe6(OH)12SO4:nH2O(s)',
    'Siderite', 'Goethite', 'Lepidocrocite', 'Rodolicoite',
    'FePO4:2H2O(s)', 'Mackinawite', 'Troilite', 'Marcasite',
    'Pyrite', 'Ferroselite', 'Glauconite', 'H4Edta(cr)',
    'Sabugalite', 'Heulandite_1', 'Heulandite_2', 'Hg2Cl2(cr)',
    'Hg3(PO4)2(s)', 'HgCO3(HgO)2(s)', 'HgHPO4(s)', 'Montroydite(red)',
    'HgS(s)', 'Ho(OH)3(cr)', 'Ho2(CO3)3(cr)', 'HoF3(cr)',
    'HoOHCO3(cr)', 'HoPO4(s)', 'Hydrosodalite', 'Illite_IMt-2',
    'Illite(Al)', 'Illite(FeII)', 'Illite(FeIII)', 'Illite(Mg)',
    'K(UO2)(AsO4):3H2O(cr)', 'K(UO2)(BO3):H2O(cr)', 'K(UO2)(PO4):3H2O(cr)', 'Boltwoodite',
    'Compreignacite', 'K3NpO2(CO3)2(s)', 'K4NpO2(CO3)3(s)', 'KNpO2CO3(s)',
    'Li(UO2)(AsO4):4H2O(cr)', 'Li(UO2)(BO3):1.5H2O(cr)', 'Li(UO2)(PO4):4H2O(cr)', 'Linda_type_A',
    'Low-silica_P-Ca', 'Low-silica_P-Na', 'Brucite', 'Glushinskite',
    'Mg[(UO2)(AsO4)]2:10H2O(cr)', 'Meta-saleite', 'Mg2KH(PO4)2:15H2O(cr)', 'Farringtonite',
    'Cattite', 'Mg3(PO4)2:4H2O(cr)', 'Bobierrite', 'Magnesite',
    'Newberyite', 'Phosphorröslerite', 'MgKPO4:H2O(cr)', 'K-struvite',
    'MgSeO3:6H2O(cr)', 'Pyrochroite', 'Mn[(UO2)(AsO4)]2:8H2O(cr)', 'Mn[(UO2)(PO4)]2:10H2O(cr)',
    'Bixbyite', 'Hausmannite', 'Rhodochrosite', 'Manganosite',
    'Manganite', 'MnSeO3:2H2O(cr)', 'Molecular_sieve_4Å', 'Montmorillonite(HcCa)',
    'Montmorillonite(HcK)', 'Montmorillonite(HcMg)', 'Montmorillonite(HcNa)', 'Montmorillonite(MgCa)',
    'Montmorillonite(MgK)', 'Montmorillonite(MgMg)', 'Montmorillonite(MgNa)', 'Molybdite',
    'Mordenite-Ca', 'Mordenite-Na', 'Na(UO2)(AsO4):3H2O(cr)', 'Na(UO2)(BO3):H2O(cr)',
    'Na(UO2)(PO4):3H2O(cr)', 'Na-boltwoodite', 'Na2Np2O7:0.1H2O(cr)', 'Na2U2O7:H2O(cr)',
    'Na3NpO2(CO3)2(cr)', 'Na6Th(CO3)5:12H2O(cr)', 'Na7HNb6O19:15H2O(cr)', 'Dawsonite',
    'NaAm(CO3)2:5H2O(cr)', 'NaAmO2CO3(s)', 'NaNpO2CO3:3.5H2O(cr)', 'Natrolite',
    'Nb2O5(pr)', 'NH4(UO2)(AsO4):3H2O(cr)', 'NH4(UO2)(PO4):3H2O(cr)', 'Ni(OH)2(cr_beta)',
    'Ni(Oxa):2H2O(cr)', 'Rauchite', 'Ni[(UO2)(PO4)]2:8H2O(cr)', 'Ni3(AsO3)2:xH2O(s)',
    'Ni[(UO2)(PO4)]2:8H2O(cr)', 'NiCO3(cr)', 'NiCO3:5.5H2O(s)', 'NiO(cr)',
    'NiSeO3:2H2O(cr)', 'Nontronite_Nau-1', 'Nontronite(Ca)', 'Nontronite(K)',
    'Nontronite(Mg)', 'Nontronite(Na)', 'Np(Oxa)2:6H2O(cr)', 'NpO2(am_hyd)',
    'NpO2(OH)2:H2O(cr_hex)', 'NpO2CO3(cr)', 'NpO2OH(am)', 'Pa2O5(act)',
    'Lead', 'Pb[(UO2)(AsO4)]2:10H2O(cr)', 'Pb[(UO2)(PO4)]2:8H2O(cr)', 'Pb2(CO3)Cl2(s)',
    'Parsonite', 'Pb3(PO4)2(s)', 'Pb5(PO4)3Cl(s)', 'PbClOH(s)',
    'PbCO3(s)', 'PbHPO4(s)', 'PbO(s_red)', 'PbO(s_yellow)',
    'PbS(s)', 'PbSO4(cr)', 'Palladium', 'PdO(cr)',
    'Phillipsite-Na', 'Phillipsite-NaK', 'Polonium', 'Po(SO4)2:H2O(s)',
    'PoO2(s)', 'PoSO4(s)', 'Pu(HPO4)2(am_hyd)', 'Pu(OH)3(am)',
    'Pu(Oxa)2:6H2O(cr)', 'Pu2(Oxa)3:10H2O(cr)', 'PuO2(am_hyd)', 'PuO2(OH)2(am_hyd)',
    'PuO2CO3(cr)', 'PuO2OH(am)', 'PuPO4(am_hyd)', 'RaCO3(cr)',
    'RaSO4(cr)', 'Ripidolite_Cca-2', 'S(orth)', 'Saponite_SapCa-2',
    'Saponite(Ca)', 'Saponite(FeCa)', 'Saponite(FeK)', 'Saponite(FeMg)',
    'Saponite(FeNa)', 'Saponite(K)', 'Saponite(Mg)', 'Saponite(Na)',
    'Scolecite', 'Selenium', 'Silica(am)', 'Quartz',
    'Sm(OH)3(cr)', 'Sm2(CO3)3(cr)', 'Smectite_MX80', 'SmF3(cr)',
    'SmOHCO3(cr)', 'Sm-rhabdophane', 'SnO(s)', 'SnO2(am)',
    'Cassiterite', 'Sodalite', 'Sr[(UO2)(AsO4)]2:8H2O(cr)', 'Sr[(UO2)(PO4)]2:6H2O(cr)',
    'Sr3(PO4)2(s)', 'Strontianite', 'SrHPO4(beta)', 'SrSeO3(cr)',
    'Celestite', 'Stilbite', 'TcO2(am_hyd_ag)', 'TcO2(am_hyd_fr)',
    'Th3(PO4)4(s)', 'ThF4(cr_hyd)', 'ThO2(am_hyd_ag)', 'ThO2(am_hyd_fr)',
    'TiO2(am_hyd)', 'U(OH)2SO4(cr)', 'U(Oxa)2:6H2O(cr)', 'UF4:2.5H2O(cr)',
    'UO2(am_hyd)', 'Rutherfordine', 'Hydrogen_uranospinite', 'Chernikovite',
    'UO2Oxa:3H2O(cr)', 'Metaschoepite', 'Coffinite', 'Vermiculite_SO',
    'Vermiculite(Ca)', 'Vermiculite(K)', 'Vermiculite(Mg)', 'Vermiculite(Na)',
    'Wülfingite', 'Zn[(UO2)(AsO4)]2:8H2O(cr)', 'Zn[(UO2)(PO4)]2:8H2O(cr)', 'Zn3(PO4)2:4H2O(s)',
    'Hydrozincite', 'Smithsonite', 'Zincite', 'Sphalerite',
    'Zr(HPO4)2:H2O(cr)', 'Zr(OH)4(am_fr)', 'Baddeleyite', 'As(cr)',
    'Graphite', 'Cu(cr)', 'Iron(alpha)', 'Pyrolusite',
    'Tugarinovite', 'Tin(beta)', 'Titanium'
]

# other_solids = ['C3AS3(cr)', 'C3FS3(cr)', 'CaSO4w0.5(cr)',  'CaSiO3(cr)', 'Al(OH)3(s)', 'Al(OOH)(alpha)', 'Al2O3(alpha)']

zeolites = [
    'Analcime', 'Cancrinite-NO3', 'Chabazite-Ca', 'Chabazite-Na', 'Clinoptilolite',
    'Faujasite-X', 'Faujasite-Y', 'Heulandite_1', 'Heulandite_2', 'Hydrosodalite',
    'Linda_type_A', 'Low-silica_P-Ca', 'Low-silica_P-Na', 'Molecular_sieve_4Å', 'Mordenite-Ca',
    'Mordenite-Na', 'Natrolite', 'Phillipsite-Na', 'Phillipsite-NaK', 'Sodalite', 'Stilbite'
]

