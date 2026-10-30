"""Split the GEMS3 backup JSON files written by ExportToGems3.ExportAllJson into several record sets.

A map file (see databases/THEREDA/scripts-out/gems3-split-map.json) describes, per group, what belongs to it:

* ``formulas``: formulas of the species of the other source (e.g. CEMDATA 18). A solid DComp / ReacDC
  record whose element composition equals one of them belongs to the group (``exclude_species`` takes out
  look-alikes such as polymorphs, by record name);
* ``phases``: exact ThermoHub phase symbols (GEMS3 ``PhName``) or fnmatch patterns, added explicitly;
* ``states``: GEMS3 phase-state letters of the DComp / ReacDC key (``a`` aqueous, ``s`` solid, ``g``/``f`` gas
  or fluid). Without ``formulas`` every record with such a state belongs to the group.

Groups are applied in the order of the map to what the earlier groups left in the main files, so a state
group such as "solids" after "cemdata" only gets the non-cement solids.

For every group:

* a phase moves when ALL its species belong to the group (or it is listed in ``phases``); a phase with only
  some species in the group stays and is reported. This covers pure phases and solid solutions with their
  end-member phases; the matching DComp / ReacDC records move with them (found through ``pDC_v_``);
* the data sources (SDref) are NOT split (see ``split_sdref``): the same key in two databases of one GEMS project is an error.

The group's records are written to ``<Record>.<group>.json`` (``Phase.<group>.pure.json`` for the pure
phases); the main files are rewritten without them. IComp is not split (the elements are shared).
"""
import collections
import fnmatch
import json
import os
import re

RECORDS = (("DComp", "dcomp"), ("ReacDC", "reacdc"), ("Phase", "phase"), ("Phase.pure", "phase"), ("SDref", "sdref"))
_FILES = {"DComp": "DComp.backup.json", "ReacDC": "ReacDC.backup.json", "Phase": "Phase.backup.json",
          "Phase.pure": "Phase.backup.pure.json", "SDref": "SDref.backup.json"}


def composition(formula):
    """Element composition of a GEMS formula as a hashable tuple (oxidation-state markers, charge and
    ``@`` ignored, parentheses/brackets/braces and decimal coefficients supported)."""
    f = re.sub(r"\|[+-]?\d+\|", "", formula).strip()
    f = f.replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")").replace(":", "")
    f = re.sub(r"@$", "", f)
    if re.search(r"[A-Za-z)][+-]\d*$", f):
        f = re.sub(r"[+-]\d*$", "", f)
    pos = 0

    def number():
        nonlocal pos
        m = re.match(r"\d+(\.\d+)?", f[pos:])
        if m:
            pos += m.end()
            return float(m.group())
        return 1.0

    def group():
        nonlocal pos
        comp = collections.Counter()
        while pos < len(f):
            c = f[pos]
            if c == "(":
                pos += 1
                inner = group()
                n = number()
                for k, v in inner.items():
                    comp[k] += v * n
            elif c == ")":
                pos += 1
                return comp
            else:
                m = re.match(r"[A-Z][a-z]?", f[pos:])
                if not m:
                    pos += 1
                    continue
                pos += m.end()
                comp[m.group()] += number()
        return comp

    return tuple(sorted((k, round(v, 3)) for k, v in group().items() if abs(v) > 1e-9))


def _formula(record):
    return record["dod"].get("RDform") or record["dod"].get("DCform") or ""


def _name(record):
    return record["dod"].get("RDname") or record["dod"].get("DCname") or ""


def _norm(text):
    return "".join(str(text).split())


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dump(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=4, ensure_ascii=False)


def _dc_key(record):
    return _norm("".join(str(part) for part in record["key"]))


def _phase_dcs(record):
    return {_norm(row[0]) for row in (record["dod"].get("pDC_v_") or [])}


def _ref_labels(record):
    labels = set()
    for field in ("dSDref", "rSDref", "pSDref"):
        for row in record["dod"].get(field) or []:
            labels.add(str(row[0]))
    return labels


def _sdref_label(record):
    return ":".join(str(part).rstrip("_") for part in record["key"])


def split_gems_records(map_file, directory, log=print):
    """Split the backup files in ``directory`` as described in the module docstring.

    Returns a list of dicts ``{"group", "tag_infix", "files": [(json_name, keyword, tag_suffix), ...]}`` for
    the groups that got records, to be converted to .pdb/.ndx (tag_suffix is "" or ".pure").
    """
    with open(map_file, encoding="utf-8") as fh:
        groups = json.load(fh)["groups"]
    data = {name: _load(os.path.join(directory, file)) for name, file in _FILES.items()
            if os.path.exists(os.path.join(directory, file))}
    result = []
    for group, spec in groups.items():
        patterns = spec.get("phases", [])
        states = spec.get("states", ["s"])
        exclude = set(spec.get("exclude_species", []))
        formulas = spec.get("formulas", [])
        formulas = list(formulas.keys()) if isinstance(formulas, dict) else formulas
        wanted = {composition(f) for f in formulas}
        if not formulas and not patterns and "states" not in spec:
            raise ValueError(f"group '{group}' needs formulas, phases or states")
        # species of the group: solid records with the composition of a listed formula
        group_species = {}
        for name in ("DComp", "ReacDC"):
            for rec in data.get(name, []):
                if (rec["key"][0] in states and _name(rec) not in exclude
                        and (not formulas or composition(_formula(rec)) in wanted)):
                    group_species[_dc_key(rec)] = rec
        all_dcs = {name: [_phase_dcs(r) for r in data.get(name, [])] for name in ("Phase", "Phase.pure")}
        chosen = {name: [] for name in ("Phase", "Phase.pure")}
        group_dcs = set()
        partial = {}
        for name in chosen:
            for rec in data.get(name, []):
                dcs = _phase_dcs(rec)
                listed = any(fnmatch.fnmatchcase(rec["dod"]["PhName"], p) for p in patterns)
                if listed or (dcs and dcs <= set(group_species)):
                    chosen[name].append(rec)
                    group_dcs |= dcs
                elif dcs & set(group_species):
                    partial[rec["dod"]["PhName"]] = sorted(dcs & set(group_species))
        # species that no phase record contains (e.g. gases: the gas phase is not exported) move on their own
        used_by_phase = set()
        for name in ("Phase", "Phase.pure"):
            for rec in data.get(name, []):
                used_by_phase |= _phase_dcs(rec)
        group_dcs |= set(group_species) - used_by_phase
        unmatched = [p for p in patterns if not any(fnmatch.fnmatchcase(r["dod"]["PhName"], p)
                                                    for n in chosen for r in data.get(n, []))]
        # pure phases consisting only of species of the group (end-members of its solid solutions)
        for name in ("Phase", "Phase.pure"):
            for rec in data.get(name, []):
                dcs = _phase_dcs(rec)
                if rec not in chosen[name] and dcs and dcs <= group_dcs:
                    chosen[name].append(rec)
                    partial.pop(rec["dod"]["PhName"], None)
        moved_phase_ids = {name: {id(r) for r in chosen[name]} for name in chosen}
        # species still used by a phase that stays behind
        staying = set()
        for name in chosen:
            for rec in data.get(name, []):
                if id(rec) not in moved_phase_ids[name]:
                    staying |= _phase_dcs(rec) & group_dcs
        moved = {"Phase": chosen["Phase"], "Phase.pure": chosen["Phase.pure"]}
        for name in ("DComp", "ReacDC"):
            moved[name] = [r for r in data.get(name, []) if _dc_key(r) in group_dcs]
        found = {_dc_key(r) for name in ("DComp", "ReacDC") for r in moved[name]}
        missing = sorted(group_dcs - found)
        labels = set()
        for name in ("DComp", "ReacDC", "Phase", "Phase.pure"):
            for rec in moved[name]:
                labels |= _ref_labels(rec)
        # A data source must exist in ONE database only: GEMS refuses to load two databases of a project
        # that hold the same SDref key ("Two records with the same key"). So by default the groups get no
        # SDref file of their own and every cited source stays in the single main SDref set; set
        # "split_sdref": true in a group to copy the sources it cites into its own file (duplicates!).
        moved["SDref"] = ([r for r in data.get("SDref", []) if _sdref_label(r) in labels]
                          if spec.get("split_sdref", False) else [])
        n_total = sum(len(v) for k, v in moved.items() if k != "SDref")
        log(f"gems3_split group '{group}': " + ", ".join(f"{len(moved[n])} {n}" for n in moved))
        if unmatched:
            log(f"  WARNING: phases in the map that do not exist in this export: {unmatched}")
        if partial:
            log(f"  WARNING: phases with only some species matching the group formulas (kept in the main set): {partial}")
        orphan = sorted(_name(r) for k, r in group_species.items() if k not in group_dcs)
        if orphan:
            log(f"  WARNING: species matching the group formulas that are in no moved phase (kept in the main set): {orphan}")
        if missing:
            log(f"  WARNING: species of the group phases without a DComp/ReacDC record: {missing}")
        if staying:
            log(f"  WARNING: species of the group still used by phases outside it: {sorted(staying)}")
        if not n_total:
            continue
        files = []
        for name, keyword in RECORDS:
            if not moved.get(name):
                continue
            suffix = ".pure" if name == "Phase.pure" else ""
            base = "Phase" if name == "Phase.pure" else name
            out = f"{base}.{group}{suffix}.json"
            _dump(os.path.join(directory, out), moved[name])
            files.append((out, keyword, suffix))
        # main files without the moved records (SDref stays complete: other records may cite them)
        for name in ("DComp", "ReacDC", "Phase", "Phase.pure"):
            ids = {id(r) for r in moved[name]}
            data[name] = [r for r in data[name] if id(r) not in ids]
            _dump(os.path.join(directory, _FILES[name]), data[name])
        result.append(dict(group=group, tag_infix=spec.get("tag_infix", group), files=files))
    return result
