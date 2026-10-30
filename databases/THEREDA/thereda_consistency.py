#!/usr/bin/env python
"""Consistency checker (and optional fixer) for THEREDA_2026-01_JSON.json.

Standalone checker adapted from thereda_consistency_check_handoff/; the import notebook calls
check_and_fix() before runImport. Fix mode "prefer-logK" is described at fix_prefer_logk().

Walks every species record (PrimaryMaster / SecondaryMaster / Product /
MineralsSolids in every phase) and checks, at 298.15 K / 1.01325 bar:

  REACTANTS   every FormingReaction reactant exists in the database
  ELEMENTS    forming reaction is element-balanced
  CHARGE      forming reaction is charge-balanced; declared charge matches
  MOLARMASS   declared molar mass equals sum of element masses
  LOGK_DRG    DRG298 = -RT ln10 * LOGK298
  DFG_DRG     DFG298 = DRG298 + sum(nu_i * DFG298_i)   (reactants)
  DFH_DRH     DFH298 = DRH298 + sum(nu_i * DFH298_i)
  S_DRS       S298   = DRS298 + sum(nu_i * S298_i)
  CP_DRCP     CP298  = DRCP298 + sum(nu_i * CP298_i)
  DRG_DRH_DRS DRG298 = DRH298 - T * DRS298
  DFG_DFH_S   DFG298 = DFH298 - T * (S298 - sum(n_el * S298_el))
  MISSING     DFG298 / DRG298 / LOGK298 absent (WARN; says whether it could be derived)
  DUPLICATE   same symbol defined more than once with different data

Usage:
  python check_thereda_consistency.py [json] [--out report.tsv] [--symbol X]
         [--only CHECK[,CHECK]] [--tol-g 1.0] [--tol-logk 0.005]

All findings are warnings (the data are imported as given); the exit status is always 0.
"""
import argparse
import collections
import json
import math
import os
import sys

R = 8.314462618
T = 298.15
RT_LN10 = R * T * math.log(10)

SPECIES_LISTS = ("PrimaryMaster", "SecondaryMaster", "Product", "MineralsSolids")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, "data-in", "2026", "THEREDA_2026-01_JSON.json")
DEFAULT_OUT = os.path.join(HERE, "data-out", "consistency", "thereda_consistency_report.tsv")


def load_species(db):
    """Return list of (phase, listname, record) for every species."""
    out = []
    for ph in db["Phases"]:
        for ln in SPECIES_LISTS:
            for rec in ph.get(ln) or []:
                out.append((ph.get("symbol"), ln, rec))
    return out


def std(rec):
    """datatype -> value from DataStandard (None when absent/null)."""
    return {x["datatype"]: x.get("value") for x in rec.get("DataStandard", [])}


def mean_diff(vals):
    return max(vals) - min(vals)


class Report:
    def __init__(self):
        self.rows = []
        self.evaluated = collections.Counter()  # check -> number of records actually tested
        self.comparisons = []  # every numeric stored-vs-calculated comparison made

    def compare(self, check, symbol, prop, unit, stored, calc, tol, sev="WARN", rel=False, detail=""):
        """Record one stored-vs-calculated comparison; add a finding if it exceeds tol."""
        self.evaluated[check] += 1
        diff = stored - calc
        bad = abs(diff) > (tol * abs(calc) if rel else tol)
        self.comparisons.append(dict(check=check, symbol=symbol, property=prop, unit=unit,
                                     stored=stored, calc=calc, diff=diff, detail=detail, tol=tol,
                                     status=sev if bad else "ok"))
        if bad:
            self.add(sev, check, symbol, f"{prop} stored {stored:.6g} vs calc {calc:.6g} {unit}",
                     property=prop, unit=unit, stored=round(stored, 6), implied=round(calc, 6),
                     diff=round(diff, 6))

    def add(self, sev, check, symbol, msg, **kw):
        self.rows.append(dict(severity=sev, check=check, symbol=symbol, message=msg, **kw))


def run_checks(db, tol_g, tol_logk, tol_mass, tol_el, tol_v=0.01):
    rep = Report()
    elements = {e["symbol"]: e for e in db["Elements"]}
    species = load_species(db)

    # first definition of each symbol wins as lookup; duplicates are checked separately
    by_sym = {}
    dupes = collections.defaultdict(list)
    for ph, ln, rec in species:
        s = rec["symbol"]
        dupes[s].append(rec)
        by_sym.setdefault(s, rec)
    for s, recs in dupes.items():
        if len(recs) > 1:
            sigs = {json.dumps([std(r), r.get("FormingReaction")], sort_keys=True) for r in recs}
            if len(sigs) > 1:
                rep.add("WARN", "DUPLICATE", s, f"{len(recs)} definitions with differing data")

    def dec(rec, key):
        return (rec.get("Declaration") or {}).get(key)

    seen = set()
    for ph, ln, rec in species:
        sym = rec["symbol"]
        if sym in seen:
            continue
        seen.add(sym)
        sd = std(rec)
        comp = {c["element"]: c["numberofelement"] for c in rec.get("Composition", [])}
        fr = rec.get("FormingReaction") or []

        # --- declared molar mass vs composition ---
        mm = dec(rec, "molarmass")
        if mm is not None and comp:
            try:
                calc = sum(n * elements[e]["molarmass"] for e, n in comp.items())
                rep.compare("MOLARMASS", sym, "molarmass", "g/mol", mm, calc, tol_mass,
                            detail="composition: " + ", ".join(f"{n:g} {e}" for e, n in comp.items()))
            except KeyError as ex:
                rep.add("WARN", "MOLARMASS", sym, f"element {ex} missing from Elements")

        # --- missing Gibbs energy data (WARN only; nothing is filled in here) ---
        has_reaction = len([r for r in fr if r["pcon_reactant"] != sym]) > 0
        dfg_, drg_, lk_ = sd.get("DFG298"), sd.get("DRG298"), sd.get("LOGK298")
        if has_reaction:
            if (drg_ is None) != (lk_ is None):
                have, lack = ("DRG298", "LOGK298") if lk_ is None else ("LOGK298", "DRG298")
                rep.add("WARN", "MISSING", sym, f"{lack} missing, derivable from {have}")
            elif drg_ is None:
                rep.add("WARN", "MISSING", sym, "neither DRG298 nor LOGK298 given for a species "
                        "defined by a reaction")
            if dfg_ is None:
                reacts_ok = all(std(by_sym[r["pcon_reactant"]]).get("DFG298") is not None
                                for r in fr if r["pcon_reactant"] != sym and r["pcon_reactant"] in by_sym)
                how = ("derivable from DRG298/LOGK298 and the reactants' DFG298"
                       if (drg_ is not None or lk_ is not None) and reacts_ok
                       else "not derivable (no reaction G, or a reactant lacks DFG298)")
                rep.add("WARN", "MISSING", sym, f"DFG298 missing, {how}")
        elif dfg_ is None:
            rep.add("WARN", "MISSING", sym, "DFG298 missing (species without a forming reaction)")

        # --- reaction checks ---
        reactants = [r for r in fr if r["pcon_reactant"] != sym]
        selfc = [r for r in fr if r["pcon_reactant"] == sym]
        if fr and (len(selfc) != 1 or selfc[0]["coefficient"] != 1):
            rep.add("WARN", "REACTION", sym,
                    "forming reaction does not contain the species exactly once with coefficient +1")
        missing = [r["pcon_reactant"] for r in reactants if r["pcon_reactant"] not in by_sym]
        for m in missing:
            rep.add("WARN", "REACTANTS", sym, f"reactant '{m}' not found in database")

        if fr and not missing:
            # element + charge balance (species itself counted with its own coefficient)
            tot = collections.Counter()
            chg = 0.0
            for r in fr:
                rr = by_sym[r["pcon_reactant"]] if r["pcon_reactant"] != sym else rec
                for c in rr.get("Composition", []):
                    tot[c["element"]] += r["coefficient"] * c["numberofelement"]
                chg += r["coefficient"] * (dec(rr, "charge") or 0)
            bad = {e: v for e, v in tot.items() if abs(v) > 1e-9 and e != "EA"}
            if bad:
                rep.add("WARN", "ELEMENTS", sym, f"reaction not element-balanced: {bad}")
            if abs(chg) > 1e-9:
                rep.add("WARN", "CHARGE", sym, f"reaction charge imbalance {chg:+g}")

            def sum_nu(key):
                vals = [by_sym[r["pcon_reactant"]] for r in reactants]
                tot_ = 0.0
                for r, rr in zip(reactants, vals):
                    v = std(rr).get(key)
                    if v is None:
                        return None
                    tot_ += r["coefficient"] * v
                return tot_

            # Reaction properties from the same record: "gibbs-like" sign convention
            # DRX = X_species + sum(nu_reactant * X_reactant)  (nu_reactant is negative)
            for prop, rprop, label, unit in (("DFG298", "DRG298", "DFG_DRG", "J/mol"),
                                             ("DFH298", "DRH298", "DFH_DRH", "J/mol"),
                                             ("S298", "DRS298", "S_DRS", "J/mol/K"),
                                             ("CP298", "DRCP298", "CP_DRCP", "J/mol/K")):
                f, d = sd.get(prop), sd.get(rprop)
                s = sum_nu(prop)
                if None in (f, d, s):
                    continue
                tol = tol_g if prop in ("DFG298", "DFH298") else tol_g / 10
                terms = "; ".join(f"{r['coefficient']:+g} x {r['pcon_reactant']} ({prop} = "
                                  f"{std(by_sym[r['pcon_reactant']]).get(prop):.8g})" for r in reactants)
                rep.compare(label, sym, f"{prop} (calc = {rprop} - sum(nu*{prop}_react))",
                            unit, f, d - s, tol,
                            detail=f"{rprop} of the reaction = {d:.8g}; reactants: {terms}")

        # --- logK vs DRG ---
        lk, drg = sd.get("LOGK298"), sd.get("DRG298")
        if lk is not None and drg is not None:
            rep.compare("LOGK_DRG", sym, "LOGK298 (calc = -DRG298/RTln10)", "-", lk,
                        -drg / RT_LN10, tol_logk, detail=f"DRG298 = {drg:.8g} J/mol")

        # --- DRG = DRH - T*DRS ---
        drh, drs = sd.get("DRH298"), sd.get("DRS298")
        if None not in (drg, drh, drs):
            rep.compare("DRG_DRH_DRS", sym, "DRG298 (calc = DRH298 - T*DRS298)", "J/mol",
                        drg, drh - T * drs, max(tol_g, 10),
                        detail=f"DRH298 = {drh:.8g} J/mol; DRS298 = {drs:.8g} J/mol/K; T = {T} K")

        # --- Gibbs-Helmholtz against element reference entropies ---
        dfg, dfh, s298 = sd.get("DFG298"), sd.get("DFH298"), sd.get("S298")
        if None not in (dfg, dfh, s298) and comp:
            try:
                s_el = 0.0
                for e, n in comp.items():
                    el = elements[e]
                    s_el += n * el["s298"] / (el.get("stoichiometriccoefficient") or 1)
                rep.compare("DFG_DFH_S", sym, "DFG298 (calc = DFH298 - T*(S298 - S_el))", "J/mol",
                            dfg, dfh - T * (s298 - s_el), tol_el, sev="WARN",
                            detail=f"DFH298 = {dfh:.8g} J/mol; S298 = {s298:.6g} J/mol/K; "
                                   f"sum of element entropies = {s_el:.6g} J/mol/K; T = {T} K")
            except (KeyError, TypeError):
                pass

        # --- molar volume vs density: V298 [cm3/mol] = M [g/mol] / RHO298 [g/cm3] ---
        v, rho = sd.get("V298"), sd.get("RHO298")
        if v is not None and rho:
            if mm:
                rep.compare("V_RHO", sym, "V298 (calc = M/RHO298)", "cm3/mol", v, mm / rho,
                            tol_v, rel=True, detail=f"M = {mm} g/mol; RHO298 = {rho} g/cm3")
        elif (v is None) != (rho is None):
            rep.add("WARN", "V_RHO", sym, f"only one of V298 ({v}) / RHO298 ({rho}) is given")

    return rep, len(seen)


# ---------------------------------------------------------------------------------------------
# Fixing
# ---------------------------------------------------------------------------------------------
FIX_MODES = ("warn-only", "prefer-logK")

# formation property -> (reaction property, tolerance factor relative to tol_g)
_FORMATION_PAIRS = (("DFG298", "DRG298", 1.0), ("DFH298", "DRH298", 1.0),
                    ("S298", "DRS298", 0.1), ("CP298", "DRCP298", 0.1))


def _datum(rec, datatype):
    for x in rec.get("DataStandard", []):
        if x.get("datatype") == datatype:
            return x
    return None


def fix_prefer_logk(db, tol_g=1.0, tol_logk=0.005, max_passes=20):
    """Make the data self-consistent keeping the REACTION side: LOGK298 > DRG298 > DRH/DRS/DRCP298.

    The formation-side values (DFG/DFH/S/CP 298) are recalculated from the reaction values and
    the reactants' own values: X298 = DRX298 - sum(nu_i * X298_i) (nu of reactants is negative),
    and DRG298 from LOGK298 when the two disagree. Only a value that is present AND disagrees
    beyond tolerance is replaced - missing values are never invented.

    Changing a species can make species that use it as a reactant inconsistent, so the pass is
    repeated until nothing changes (these are reported as further changes). The input is changed
    in place - both DataStandard[].value and DataValue[].a (the one the import reads); every
    change is also noted in the record's Declaration.description (-> ThermoHub comment) and in
    the changed datum's own "description". Returns the list of changes.
    """
    changes = []
    species = load_species(db)
    records = collections.defaultdict(list)
    for _, _, rec in species:
        records[rec["symbol"]].append(rec)
    by_sym = {s: recs[0] for s, recs in records.items()}

    def set_value(sym, prop, new, check, because):
        old = None
        for rec in records[sym]:
            d = _datum(rec, prop)
            if d is None:
                continue
            old = d["value"]
            d["value"] = new
            note = (f"{prop} recalculated ({check}, prefer-logK): {old:.10g} -> {new:.10g} "
                    f"[{because}]")
            d["description"] = (d.get("description") + "; " if d.get("description") else "") + note
            # the import reads DataValue[].a (DataStandard is the 298.15 K summary of the same data)
            for dv in rec.get("DataValue", []):
                if dv.get("datatype") == prop:
                    dv["a"] = new
                    dv["description"] = (dv.get("description") + "; " if dv.get("description") else "") + note
            decl = rec.setdefault("Declaration", {})
            decl["description"] = (decl.get("description") + "; " if decl.get("description") else "") + note
        changes.append(dict(symbol=sym, property=prop, check=check, old=old, new=new,
                            diff=new - old, because=because))

    for npass in range(1, max_passes + 1):
        n_before = len(changes)
        for sym, rec in by_sym.items():
            sd = std(rec)
            # logK is kept; DRG298 follows it
            lk, drg = sd.get("LOGK298"), sd.get("DRG298")
            if lk is not None and drg is not None and abs(lk + drg / RT_LN10) > tol_logk:
                set_value(sym, "DRG298", -RT_LN10 * lk, "LOGK_DRG", "kept LOGK298")
            fr = rec.get("FormingReaction") or []
            reactants = [r for r in fr if r["pcon_reactant"] != sym]
            if not fr or any(r["pcon_reactant"] not in by_sym for r in reactants):
                continue
            sd = std(rec)
            for prop, rprop, fac in _FORMATION_PAIRS:
                f, d = sd.get(prop), sd.get(rprop)
                if f is None or d is None:
                    continue
                s = 0.0
                for r in reactants:
                    v = std(by_sym[r["pcon_reactant"]]).get(prop)
                    if v is None:
                        s = None
                        break
                    s += r["coefficient"] * v
                if s is None:
                    continue
                if abs(f - (d - s)) > tol_g * fac:
                    set_value(sym, prop, d - s, f"{prop[:3] if prop[0]=='D' else prop[:2]}_{rprop[:3]}",
                              f"kept {rprop}, pass {npass}")
        if len(changes) == n_before:
            break
    return changes


_TEXT = {
    "LOGK_DRG": "logK and the reaction Gibbs energy do not match (DRG298 = -RT ln10 * LOGK298).",
    "DFG_DRG": "DFG298 does not match the forming reaction: DFG298 = DRG298 - sum(nu * DFG298 of reactants).",
    "DFH_DRH": "DFH298 does not match the forming reaction: DFH298 = DRH298 - sum(nu * DFH298 of reactants).",
    "S_DRS": "S298 does not match the forming reaction: S298 = DRS298 - sum(nu * S298 of reactants).",
    "CP_DRCP": "CP298 does not match the forming reaction: CP298 = DRCP298 - sum(nu * CP298 of reactants).",
    "DRG_DRH_DRS": "DRG298 does not match DRH298 and DRS298 (DRG = DRH - T*DRS).",
    "DFG_DFH_S": "DFG298 does not match DFH298 and S298 (DFG = DFH - T*(S - sum of element entropies)).",
    "MOLARMASS": "Declared molar mass does not match the composition.",
    "V_RHO": "V298 does not match molar mass / density.",
}


def write_text_report(path, source_name, n_species, rep, changes=(), tols=None):
    """Plain-text report, one block per species, meant to be sent to the data providers."""
    bad = [c for c in rep.comparisons if c["status"] != "ok"]
    other = [r for r in rep.rows if "stored" not in r]
    by = collections.defaultdict(lambda: dict(cmp=[], other=[]))
    for c in bad:
        by[c["symbol"]]["cmp"].append(c)
    for r in other:
        by[r["symbol"]]["other"].append(r)
    nwarn = sum(1 for r in rep.rows if r["severity"] == "WARN")
    lines = [f"THEREDA data consistency report - {source_name}",
             f"{n_species} species checked at 298.15 K: {nwarn} warnings, "
             f"{len(by)} species affected.",
             "Values are as given in the data file (J/mol, J/mol/K, cm3/mol, g/mol; logK dimensionless). "
             "'Implied' = the value the other data of the same species require.",
             "Only properties present in the file can be checked; missing data are listed under MISSING.",
             "Forming reactions: reactants have negative coefficients, the species itself +1.", ""]
    order = sorted(by, key=lambda s: (not by[s]["cmp"], s.lower()))
    for sym in order:
        lines.append("=" * 78)
        lines.append(sym)
        for c in sorted(by[sym]["cmp"], key=lambda c: (c["status"], c["check"])):
            what = c["property"].split(" (")[0]
            lines.append(f"  [{c['status']}] {_TEXT.get(c['check'], c['check'])}")
            lines.append(f"      {what} given   : {fmt(c['stored'])} {c['unit']}")
            lines.append(f"      {what} implied : {fmt(c['calc'])} {c['unit']}")
            lines.append(f"      difference     : {c['diff']:+.6g} {c['unit']}")
            if c.get("detail"):
                lines.append(f"      from           : {c['detail']}")
        for r in by[sym]["other"]:
            lines.append(f"  [{r['severity']}] {r['check']}: {r['message']}")
    if changes:
        lines += ["", "=" * 78, "Values recalculated during import (prefer-logK):"]
        for ch in changes:
            lines.append(f"  {ch['symbol']}: {ch['property']} {ch['old']:.10g} -> {ch['new']:.10g} "
                         f"({ch['diff']:+.6g})")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def check_and_fix(json_in, json_out=None, mode="warn-only", report_tsv=None, changes_tsv=None, report_txt=None,
                  tol_g=1.0, tol_logk=0.005, tol_mass=0.01, tol_el=50.0, tol_v=0.01, verbose=True):
    """Check json_in; for mode "prefer-logK" also write the corrected copy to json_out.

    Prints one line per mismatch plus a summary; returns (import_file, report, changes), where
    import_file is json_out if the data were changed, else json_in.
    """
    if mode not in FIX_MODES:
        raise ValueError(f"mode must be one of {FIX_MODES}, not {mode!r}")
    with open(json_in, encoding="utf-8") as fh:
        db = json.load(fh)
    rep, n = run_checks(db, tol_g, tol_logk, tol_mass, tol_el, tol_v)
    bad = [c for c in rep.comparisons if c["status"] != "ok"]
    other = [r for r in rep.rows if "stored" not in r]
    if verbose:
        print(f"THEREDA consistency check: {n} species, "
              f"{sum(1 for r in rep.rows if r['severity']=='WARN')} warnings")
        for c in sorted(bad, key=lambda c: (c["status"], c["check"], c["symbol"])):
            print(f"  [{c['status']}] {c['check']:12s} {c['symbol']}: {c['property']} stored "
                  f"{c['stored']:.8g} vs calc {c['calc']:.8g} {c['unit']} (diff {c['diff']:+.6g})")
        for r in other:
            print(f"  [{r['severity']}] {r['check']:12s} {r['symbol']}: {r['message']}")
    if report_tsv:
        os.makedirs(os.path.dirname(os.path.abspath(report_tsv)), exist_ok=True)
        cols = ["status", "check", "symbol", "property", "unit", "stored", "calc", "diff"]
        with open(report_tsv, "w", encoding="utf-8", newline="") as fh:
            fh.write("\t".join(cols) + "\tnote\n")
            for c in sorted(rep.comparisons, key=lambda c: (c["status"] == "ok", c["check"], c["symbol"])):
                fh.write("\t".join(fmt(c[k]) for k in cols) + "\t\n")
            for r in other:
                fh.write(f"{r['severity']}\t{r['check']}\t{r['symbol']}\t\t\t\t\t{r['message']}\n")
    changes = []
    import_file = json_in
    if mode == "prefer-logK":
        changes = fix_prefer_logk(db, tol_g, tol_logk)
        if changes:
            if not json_out:
                raise ValueError("json_out is required for a fix mode")
            with open(json_out, "w", encoding="utf-8") as fh:
                json.dump(db, fh, ensure_ascii=False, indent=1)
            import_file = json_out
        if verbose:
            print(f"prefer-logK: {len(changes)} value(s) recalculated"
                  + (f", corrected copy written to {json_out}" if changes else ""))
            for ch in changes:
                print(f"  fixed {ch['symbol']}: {ch['property']} {ch['old']:.10g} -> {ch['new']:.10g} "
                      f"(diff {ch['diff']:+.6g}; {ch['check']}, {ch['because']})")
    if changes_tsv and changes:
        os.makedirs(os.path.dirname(os.path.abspath(changes_tsv)), exist_ok=True)
        with open(changes_tsv, "w", encoding="utf-8", newline="") as fh:
            fh.write("symbol\tproperty\tcheck\told\tnew\tdiff\tbecause\n")
            for ch in changes:
                fh.write(f"{ch['symbol']}\t{ch['property']}\t{ch['check']}\t{ch['old']!r}\t"
                         f"{ch['new']!r}\t{ch['diff']!r}\t{ch['because']}\n")
    if report_txt:
        write_text_report(report_txt, os.path.basename(json_in), n, rep, changes)
        if verbose:
            print(f"Text report for the data providers: {os.path.abspath(report_txt)}")
    return import_file, rep, changes


def fmt(x):
    if not isinstance(x, (int, float)):
        return str(x)
    return f"{x:.3f}" if abs(x) >= 1 else f"{x:.6g}"


def print_table(rows, title):
    """rows: dicts with symbol, property, unit, stored, calc, diff, status."""
    print(f"\n{title}")
    hdr = ("species", "property", "unit", "stored", "calc", "diff", "status")
    data = [(r["symbol"], r["property"], r["unit"], fmt(r["stored"]), fmt(r["calc"]),
             fmt(r["diff"]), r["status"]) for r in rows]
    w = [max(len(h), *(len(d[i]) for d in data)) if data else len(h) for i, h in enumerate(hdr)]
    w[0] = min(w[0], 60)
    w[1] = min(w[1], 55)
    line = "  ".join(h.ljust(w[i]) if i < 3 or i == 6 else h.rjust(w[i]) for i, h in enumerate(hdr))
    print(line)
    print("-" * len(line))
    for d in data:
        print("  ".join((d[i][: w[i]]).ljust(w[i]) if i < 3 or i == 6 else d[i].rjust(w[i])
                        for i in range(len(hdr))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="?", default=DEFAULT_JSON)
    ap.add_argument("--out", default=DEFAULT_OUT, help="TSV report path ('' to skip)")
    ap.add_argument("--symbol", help="only report findings for this species symbol")
    ap.add_argument("--only", help="comma list of check names to report")
    ap.add_argument("--tol-g", type=float, default=1.0, help="J/mol tolerance for G/H identities")
    ap.add_argument("--tol-logk", type=float, default=0.005)
    ap.add_argument("--tol-mass", type=float, default=0.01, help="g/mol")
    ap.add_argument("--tol-el", type=float, default=50.0, help="J/mol for DFG vs DFH-TS (WARN)")
    ap.add_argument("--tol-v", type=float, default=0.01, help="relative tolerance for V298 vs M/RHO298")
    ap.add_argument("--fix", choices=FIX_MODES, default="warn-only",
                    help="warn-only (default) or prefer-logK: write a corrected copy to --fixed-json")
    ap.add_argument("--fixed-json", default=None, help="corrected copy (default: <json>_checked.json)")
    ap.add_argument("--all", action="store_true", help="also print every comparison, not just mismatches")
    a = ap.parse_args(argv)

    if a.fix != "warn-only":
        out = a.fixed_json or os.path.splitext(a.json)[0] + "_checked.json"
        check_and_fix(a.json, out, a.fix, a.out or None,
                      os.path.join(os.path.dirname(os.path.abspath(a.out or out)), "thereda_consistency_changes.tsv"),
                      a.tol_g, a.tol_logk, a.tol_mass, a.tol_el, a.tol_v)
        return 0
    with open(a.json, encoding="utf-8") as fh:
        db = json.load(fh)
    rep, n = run_checks(db, a.tol_g, a.tol_logk, a.tol_mass, a.tol_el, a.tol_v)
    rows = rep.rows
    comps = rep.comparisons
    if a.symbol:
        rows = [r for r in rows if r["symbol"] == a.symbol]
        comps = [c for c in comps if c["symbol"] == a.symbol]
    if a.only:
        keep = set(a.only.split(","))
        rows = [r for r in rows if r["check"] in keep]
        comps = [c for c in comps if c["check"] in keep]

    # stdout may be cp1252 on Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(f"Checked {n} unique species from {os.path.basename(a.json)}")
    print("Records actually tested per property check (others lacked data on one side):")
    nwarn = collections.Counter(r["check"] for r in rep.rows if r["severity"] == "WARN")
    all_checks = ["MOLARMASS", "LOGK_DRG", "DFG_DRG", "DFH_DRH", "S_DRS", "CP_DRCP",
                  "DRG_DRH_DRS", "DFG_DFH_S", "V_RHO"]
    nerr = collections.Counter(r["check"] for r in rep.rows if r["severity"] == "WARN")
    for c in all_checks:
        print(f"  {c:12s} tested {rep.evaluated[c]:4d}   warnings {nerr[c]:3d}")
    print(f"  {'MISSING':12s} warnings {nwarn['MISSING']:3d}")

    # numeric findings: species / property / stored vs calculated
    numeric = [c for c in comps if c["status"] != "ok"]
    numeric.sort(key=lambda c: (c["status"], c["check"], c["symbol"]))
    if numeric:
        print_table(numeric, "Mismatches: stored vs calculated")
    # structural findings without a stored/calc pair
    other = [r for r in rows if "stored" not in r]
    for r in other:
        print(f"\n[{r['severity']}] {r['check']} {r['symbol']}: {r['message']}")
    if not rows:
        print("\nNo findings.")
    if a.all:
        print_table(sorted(comps, key=lambda c: (c["check"], c["symbol"])),
                    "All comparisons (stored vs calculated)")

    warns = sum(1 for r in rows if r["severity"] == "WARN")
    print(f"\nTotal: {warns} warnings")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        cols = ["status", "check", "symbol", "property", "unit", "stored", "calc", "diff"]
        with open(a.out, "w", encoding="utf-8", newline="") as fh:
            fh.write("\t".join(cols) + "\tnote\n")
            for c in sorted(comps, key=lambda c: (c["status"] == "ok", c["check"], c["symbol"])):
                fh.write("\t".join(fmt(c[k]) for k in cols) + "\t\n")
            for r in other:
                fh.write(f"{r['severity']}\t{r['check']}\t{r['symbol']}\t\t\t\t\t{r['message']}\n")
        print(f"Report (all comparisons, mismatches first) written to {os.path.abspath(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
