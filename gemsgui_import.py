# Python wrapper around GEMSGUI's `json2db` CLI tool (JSON -> GEM-Selektor
# .pdb/.ndx database file converter). See conda-install-dependencies.sh for
# how json2db gets built and installed into this conda env.
#
# json2db is a compiled binary, not a Python module or a jsonio/thermomatch
# extension -- this wrapper just shells out to it via subprocess (see the
# "Python API vs subprocess" discussion this module grew out of: json2db's
# own state is global-singleton-based C++ (TVisor/pVisor), which makes a
# real in-process Python binding needlessly complex for something that only
# needs to run a handful of times per notebook, once per JSON file).
#
# Usage from any databases/<X>/ script or notebook, e.g.
# databases/THEREDA/export-gems3.py, right after producing *.backup.json:
#
#     import sys, os
#     sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
#     from gemsgui_import import json_to_db
#
#     json_to_db("data-out/IComp.backup.json", keyword="icomp",
#                output_dir="data-out/gems-auto", tag="THEREDA.ver01-2026-r17")
#
# json_file/output_dir may be relative to your own script's cwd -- this
# module resolves json2db's own binary/-s/-u paths from CONDA_PREFIX, not
# from the caller's location, so it works the same from any databases/*/
# folder.
import os
import shutil
import subprocess

KEYWORDS = ("icomp", "dcomp", "phase", "reacdc", "sdref")


def _conda_prefix():
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        raise RuntimeError(
            "CONDA_PREFIX is not set -- run this from inside an activated "
            "'thermoimpex' conda environment (conda activate thermoimpex), "
            "see TECHINFO.md."
        )
    return prefix


def json_to_db(json_file, keyword, output_dir, tag):
    """Convert a JSON records file into a GEM-Selektor .pdb/.ndx pair.

    json_file: path to a JSON file holding an array of records (the format
        thermomatch's ExportToGems3(...).ExportAllJson(...) produces, e.g.
        databases/THEREDA/data-out/IComp.backup.json).
    keyword: which GEM-Selektor keyword the records belong to -- one of
        icomp, dcomp, phase, reacdc, sdref.
    output_dir: directory the <keyword>.<tag>.pdb/.ndx pair is written into
        (created if missing). Safe to reuse across multiple keywords/tags --
        each call only ever touches its own <keyword>.<tag>.* files.
    tag: the dotted name segment real GEM-Selektor DB files use between the
        keyword and the extension, e.g. "THEREDA.ver01-2026-r17" produces
        icomp.THEREDA.ver01-2026-r17.pdb/.ndx.

    Returns json2db's stdout (record count etc.) on success; raises
    RuntimeError with the tool's stdout/stderr on failure.
    """
    if keyword not in KEYWORDS:
        raise ValueError(f"keyword must be one of {KEYWORDS}, got {keyword!r}")

    prefix = _conda_prefix()
    binary = shutil.which("json2db") or os.path.join(prefix, "bin", "json2db")
    if not os.path.isfile(binary):
        raise FileNotFoundError(
            f"json2db not found ({binary}) -- run "
            "'bash conda-install-dependencies.sh' first."
        )

    resources_dir = os.path.join(prefix, "share", "gemsgui")
    # Pre-create so TVisor::firstTimeSetup() -- which needs a populated
    # Resources/projects/ we deliberately don't install, to keep json2db's
    # footprint ~400KB instead of Resources/'s full ~61MB -- never runs.
    # conda-install-dependencies.sh already does this once at install time;
    # repeating it here is just defense in depth (e.g. a stale/partial env).
    user_dir = os.path.join(prefix, "var", "gemsgui-user")
    os.makedirs(os.path.join(user_dir, "projects"), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        binary,
        "-j", str(json_file),
        "-k", keyword,
        "-t", tag,
        "-o", str(output_dir),
        "-s", resources_dir,
        "-u", user_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"json2db failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result.stdout
