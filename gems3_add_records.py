"""Add extra GEMS3 backup records (``IComp``/``DComp``) to an already-imported ThermoHub graph.

Two independent pieces, both used by ``databases/THEREDA/import-add-gems-records.ipynb``:

``collect_gems3_blocks()``
    turns a directory of GEMS3 backup files named ``<RecordType>.<tag>.json``
    (``IComp.N_atm.json``, ``DComp.N_atm.json``, ...) into the ``<prefix>.<block>.json``
    block files ``ReacDCImportData::splitToBlocks()`` expects, and reports, per DComp, which
    phase it belongs to (from the GEMS3 key's aggregate-state code).

``ThermoHubGraph``
    the small amount of graph wiring the declarative import cannot do, because the target
    phase/thermodataset vertices already exist and must be *extended*, not rewritten:

    * ``basis`` edge  new element -> existing thermodataset
    * ``component`` edge  new substance -> existing phase, plus the substance's symbol
      appended to that phase's ``properties.substances``

    Both matter for ``ExportToGems3``: ``select_linked()`` picks up records by traversing
    ``basis``/``belongs``/``involves``/``prodreac``/``product``/``component`` INBOUND from the
    thermodataset, and ``get_PhasePitzer()`` reads the phase's own ``properties.substances``
    list to build the GEMS3 phase's DC list.

Talks to ArangoDB over its HTTP API with nothing but the standard library, so it adds no
dependency to ``binder/environment.yml``.
"""

import base64
import glob
import json
import os
import urllib.error
import urllib.request

# GEMS3 backup record type -> import block name used in the ImpexRowDescription list.
RECORD_TYPE_BLOCKS = {
    "IComp": "elements",
    "DComp": "substances",
}

# First field of a GEMS3 DComp key is the aggregate state code (see
# ExportToGems3::elementsAggregateStateConvert): 'a' aqueous, 'g'/'f' gas/fluid, 's' solid, ...
AGGREGATE_STATE_PHASE = {
    "a": "aq",
    "g": "g",
    "f": "g",
}


def _load_records(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a GEMS3 backup json array, got {type(data).__name__}")
    return data


def collect_gems3_blocks(src_dir, prefix="add-gems", out_dir=None):
    """Merge ``<RecordType>.<tag>.json`` GEMS3 backup files into per-block import files.

    Returns a dict with ``input_file`` (what to pass to ``runImport``), ``blocks``
    (block name -> written file), ``elements`` (list of element symbols) and ``substances``
    (list of ``{"symbol", "phase", "state", "source"}``, ``phase`` being ``"aq"``/``"g"``/``None``).

    ``input_file`` is only a naming template: ``splitToBlocks()`` derives every block file from
    it by replacing the next-to-last dot-separated part with the block name, so ``prefix`` must
    not itself contain any block name as a substring.
    """
    out_dir = out_dir or src_dir
    os.makedirs(out_dir, exist_ok=True)
    for block in RECORD_TYPE_BLOCKS.values():
        if block in prefix:
            raise ValueError(f"prefix {prefix!r} must not contain the block name {block!r} - "
                             "splitToBlocks() replaces every occurrence of it in the file name")

    result = {"input_file": None, "blocks": {}, "elements": [], "substances": [], "sources": []}
    for record_type, block in RECORD_TYPE_BLOCKS.items():
        records = []
        for path in sorted(glob.glob(os.path.join(src_dir, f"{record_type}.*.json"))):
            if os.path.basename(path).startswith(prefix + "."):
                continue  # a previously generated block file, not a source drop
            found = _load_records(path)
            records.extend(found)
            result["sources"].append((os.path.relpath(path), len(found)))
        if not records:
            continue

        out_path = os.path.join(out_dir, f"{prefix}.{block}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)
            f.write("\n")
        result["blocks"][block] = out_path
        if result["input_file"] is None:
            result["input_file"] = out_path

        for record in records:
            key = record.get("key") or []
            if record_type == "IComp":
                result["elements"].append(key[0] if key else None)
            else:
                state = key[0].strip() if key else ""
                result["substances"].append({
                    "symbol": key[2] if len(key) > 2 else None,
                    "state": state,
                    "phase": AGGREGATE_STATE_PHASE.get(state),
                })
    return result


class ThermoHubGraph:
    """Minimal read/write access to a ThermoHub ArangoDB database over its HTTP API."""

    def __init__(self, url="http://localhost:8529", database="_system", user="root", password=""):
        self.url = url.rstrip("/")
        self.database = database
        self._auth = base64.b64encode(f"{user}:{password}".encode()).decode()

    def aql(self, query, **bind_vars):
        """Run one AQL query and return all result documents."""
        payload = json.dumps({"query": query, "bindVars": bind_vars, "batchSize": 1000}).encode()
        request = urllib.request.Request(
            f"{self.url}/_db/{self.database}/_api/cursor", data=payload, method="POST",
            headers={"Authorization": f"Basic {self._auth}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request) as response:
                body = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"ArangoDB query failed ({error.code}) on database "
                               f"{self.database!r}: {detail}") from None
        results = body["result"]
        while body.get("hasMore"):
            cursor_request = urllib.request.Request(
                f"{self.url}/_db/{self.database}/_api/cursor/{body['id']}", method="PUT",
                headers={"Authorization": f"Basic {self._auth}"})
            with urllib.request.urlopen(cursor_request) as response:
                body = json.load(response)
            results.extend(body["result"])
        return results

    # -- lookups ---------------------------------------------------------------------------

    def vertex_id(self, collection, symbol):
        """_id of the single vertex in `collection` with properties.symbol == symbol."""
        found = self.aql("FOR v IN @@collection FILTER v.properties.symbol == @symbol RETURN v._id",
                         **{"@collection": collection, "symbol": symbol})
        if not found:
            raise KeyError(f"no {collection} record with symbol {symbol!r} in {self.database!r}")
        if len(found) > 1:
            raise KeyError(f"{len(found)} {collection} records with symbol {symbol!r} in "
                           f"{self.database!r}: {found} - pass an explicit _id instead")
        return found[0]

    # -- edges -----------------------------------------------------------------------------

    def _upsert_edge(self, collection, from_id, to_id, properties):
        """Create the edge unless one with the same _from/_to already exists. Idempotent."""
        return self.aql(
            "UPSERT { _from: @from, _to: @to } "
            "INSERT { _from: @from, _to: @to, _type: 'edge', _label: @label, properties: @props } "
            "UPDATE {} IN @@collection "
            "RETURN { id: NEW._id, created: OLD == null }",
            **{"@collection": collection, "label": collection,
               "from": from_id, "to": to_id, "props": properties})[0]

    def add_elements_to_thermodataset(self, element_symbols, thermodataset_symbol):
        """Wire new elements into an existing thermodataset with ``basis`` edges.

        Without this the elements exist in the graph but ``select_linked(..., "element", ...)``
        never reaches them, so they are missing from the exported ``IComp.backup.json``.
        """
        thermodataset_id = self.vertex_id("thermodatasets", thermodataset_symbol)
        report = []
        for symbol in element_symbols:
            element_id = self.vertex_id("elements", symbol)
            edge = self._upsert_edge("basis", element_id, thermodataset_id, {"b_type": "element"})
            report.append({"element": symbol, "edge": edge["id"], "created": edge["created"]})
        return report

    def add_substances_to_phase(self, substance_symbols, phase_symbol):
        """Add substances to an existing phase: ``component`` edge + ``properties.substances``.

        The phase vertex is *updated in place* rather than re-imported, so everything the
        THEREDA import already put on it (mixing model, thermodataset, T/P limits, the
        substances it already had) survives.
        """
        phase_id = self.vertex_id("phases", phase_symbol)
        report = []
        for symbol in substance_symbols:
            substance_id = self.vertex_id("substances", symbol)
            class_ = self.aql("FOR s IN substances FILTER s._id == @id RETURN s.properties.class_",
                              id=substance_id)[0]
            edge = self._upsert_edge("component", substance_id, phase_id,
                                     {"pc_type": class_, "pc_conscale": {"0": "C_UNITY"}})
            report.append({"substance": symbol, "edge": edge["id"], "created": edge["created"]})

        substances = self.aql(
            "FOR p IN phases FILTER p._id == @id "
            "LET merged = APPEND(p.properties.substances, @new, true) "
            "UPDATE p WITH { properties: { substances: merged } } IN phases "
            "RETURN merged",
            id=phase_id, new=list(substance_symbols))[0]
        return {"phase": phase_symbol, "components": report, "substances": substances}
