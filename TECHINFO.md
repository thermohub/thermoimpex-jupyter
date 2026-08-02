# Technical notes

## Installing and running this repo (headless, no GUI)

The notebooks here don't use the ThermoMatch desktop app — they drive the same
underlying libraries directly from Python (`import thermomatch`, `thermofun`,
`chemicalfun`, `thermohubclient`) against a local ArangoDB-backed ThermoHub
database. This repo ships a trimmed-down copy of
[thermomatch](https://bitbucket.org/gems4/thermomatch)'s own conda setup with
the Qt GUI pieces removed:

| file | purpose |
| --- | --- |
| [`environment.devenv.yml`](environment.devenv.yml) | conda-devenv spec for the `thermoimpex` env. Same idea as thermomatch's `environment.devenv.yml`, minus `qt6-main`/`qt6-charts`/`qt6-webengine`, plus the plain-Python packages the notebooks import (`pandas`, `fuzzywuzzy`), plus `jupyterlab`. |
| [`conda-install-dependencies.sh`](conda-install-dependencies.sh) | Builds the C++/Python libraries from source: `jsonarango`, `jsonio17`, `jsonimpex17`, `ChemicalFun`, `ThermoFun`, `ThermoHubClient` (all from their bitbucket/github sources, not the conda-forge packages — needed to get an ABI-matching `thermomatch` build), then `thermomatch` itself with `-DTHERMOMATCH_APPLICATION=OFF` (GUI disabled) and `-DTHERMOMATCH_BUILD_PYTHON=ON`. Skips `jsonui17` and `thermofungui`, which thermomatch's script only builds for the desktop app. |

### 1. Prerequisites

- **Git** — `sudo apt-get install git`
- **A C/C++ toolchain + cmake** — `sudo apt-get install g++ cmake make`
- **Conda** (Miniconda or Anaconda) — if not installed:
  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  ```
- **conda-devenv**, installed into the base environment:
  ```bash
  conda install -n base -c conda-forge conda-devenv
  ```

### 2. Create and activate the `thermoimpex` conda environment

From the root of this repo:

```bash
conda devenv
conda activate thermoimpex
```

### 3. Build the C++/Python dependencies

Still inside the activated `thermoimpex` environment:

```bash
./conda-install-dependencies.sh
```

This clones and builds each dependency under `~/code` and installs into
`$CONDA_PREFIX`, then does the same for `thermomatch` itself (with the Qt
application disabled), removing the scratch source afterwards. It's
idempotent — rerun it any time; each step is skipped if its library is
already present in `$CONDA_PREFIX`.

### 4. Install ArangoDB (if not already installed)

The import/export notebooks read and write a local ArangoDB instance holding
a mirror of the `hub_main`/`resources` ThermoHub databases.

Check first whether it's already installed and running:

```bash
dpkg -l | grep arangodb3
systemctl status arangodb3
```

If not, on (K)Ubuntu/Debian, install the current version from the
[ArangoDB download page](https://www.arangodb.com/download-major/ubuntu/):

```bash
curl -OL https://download.arangodb.com/arangodb39/DEBIAN/Release.key
sudo apt-key add - < Release.key
echo 'deb https://download.arangodb.com/arangodb39/DEBIAN/ /' | sudo tee /etc/apt/sources.list.d/arangodb.list
sudo apt-get install apt-transport-https
sudo apt-get update
sudo apt-get install arangodb3
```

(On CentOS/RHEL, substitute "RPM" for "DEBIAN" and `dnf` for `apt-get`.)

Alternatively, download a specific `.deb` package directly, e.g.:

```bash
wget https://download.arangodb.com/arangodb39/DEBIAN/amd64/arangodb3_3.9.12-1_amd64.deb
sudo apt install ./arangodb3_3.9.12-1_amd64.deb
```

During setup you'll be asked a few questions — for the local db root
password you may set one or leave it empty, then confirm through the
remaining prompts (ok, ok, yes, yes).

Once installed, mirror the remote `hub_main`/`resources` ThermoHub databases
locally (script lives in the `thermomatch` repo, not here):

```bash
~/gitTHERMOMATCH/thermomatch/thermohub-local-tdb--import-from-remote.sh
```

If a root password was set above (default assumed is empty), edit that
script's `userPassword=""` (around line 98) accordingly first.

### 5. Run the notebooks

```bash
jupyter lab
```

Pick the `thermoimpex` kernel (registered automatically via `ipykernel` in
`environment.devenv.yml`) and open any `import-*.ipynb` / `export-*.ipynb`
under `databases/<NAME>/`.

## mybinder.org

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/thermohub/thermoimpex-jupyter/main?urlpath=lab)

The `binder/` folder makes this repo launchable directly from mybinder.org
(via [repo2docker](https://repo2docker.readthedocs.io/)), which auto-detects
config files there ahead of the repo root:

| file | purpose |
| --- | --- |
| `binder/environment.yml` | Same conda-forge package set as the root `environment.devenv.yml`, hand-kept in sync (repo2docker doesn't understand the conda-devenv format), without a `name:` key so it installs straight into the default kernel env. |
| `binder/apt.txt` | System build tools (`build-essential`, `cmake`, `git`, `ccache`) installed as root at image-build time. |
| `binder/postBuild` | Runs `conda-install-dependencies.sh` (same script as local setup, §3 above) to compile the `thermomatch`/`thermofun`/`chemicalfun`/`thermohubclient` stack, then downloads the ArangoDB `.deb` and extracts it with `dpkg-deb -x` — no root available at this stage, and ArangoDB has neither a conda-forge package nor an apt repo `apt.txt` can reach, so this sidesteps both. |
| `binder/start` | Launches the extracted `arangod` as a local, unauthenticated, `127.0.0.1`-only background process against a fresh empty database directory, before handing off to the Jupyter server. |

**Caveats (unverified, best-effort):** these files were written without
access to Docker or a network connection to actually run a repo2docker build,
so treat the first real launch as a shakeout run rather than a guarantee:

- The from-source compile of ~6 C++ libraries is heavy; mybinder.org's free
  build resources/time limits may not be enough (build runs at reduced
  parallelism, `THREADS=2`, to lower OOM risk, but that also makes it slower).
- The ArangoDB `.deb` is extracted rather than properly installed, so its
  shared-library dependencies aren't resolved by `apt` — if `arangod` fails
  to start, check `${HOME}/arangodb-data/arangod.log` in the session first.
- No ThermoHub data is mirrored in: the remote `hub_main`/`resources`
  databases require credentials that can't be embedded in a public Binder
  image, so sessions start against an empty local ArangoDB.

## Resources/ (git subtree from thermomatch)

`Resources/` is not maintained in this repo — it's pulled in via
[git subtree](https://git-scm.com/book/en/v2/Git-Tools-Advanced-Merging#_subtree_merge)
from the [thermomatch](https://bitbucket.org/gems4/thermomatch) repository, which owns
the schemas, config templates and other shared resources under that folder.

### Do I need to do any of this after a fresh clone?

No, not just to use the repo. `Resources/` is committed as ordinary files —
a plain `git clone` already gives you the full current content, nothing extra
required. The setup and commands below are only needed if you want to **pull in
newer changes from thermomatch**, or **push local edits under `Resources/` back
upstream**.

### Checking which thermomatch branch/commit `Resources/` currently reflects

There's no persistent tracking ref for this — the branch and source commit are
recorded in the commit message each time `Resources/` is updated (by convention:
`Add 'Resources/' from thermomatch commit <sha> (branch <branch>)`). Find the most
recent one with:

```bash
git log --grep="^Add 'Resources/'" -1
```

or, to see the full history of these imports:

```bash
git log --oneline --grep="^Add 'Resources/'" -- Resources
```

To confirm the working tree still matches that upstream commit exactly:

```bash
git fetch thermomatch <branch>
git ls-tree thermomatch/<branch> -- Resources   # expected tree hash
git ls-tree HEAD -- Resources                   # current tree hash
```

### One-time setup (only needed to pull/push)

```bash
git remote add thermomatch git@bitbucket.org:gems4/thermomatch.git
```

If `git subtree` isn't available (`git: 'subtree' is not a git command`), your git
install is missing the contrib script from its exec-path
(`git --exec-path`). Copy `git-subtree` from any recent git installation's
`libexec/git-core/` (or `/usr/lib/git-core/`) into that directory and `chmod +x` it.

### Pulling upstream changes into `Resources/`

```bash
git fetch thermomatch <branch>
git subtree pull --prefix=Resources thermomatch <branch> -m "Add 'Resources/' from thermomatch commit <sha> (branch <branch>)"
```

Gotchas:
- `git subtree pull` runs a real `git merge` and requires a **fully clean working
  tree** (not just `Resources/`) — `git stash push -u` first if you have unrelated
  work in progress, then `git stash pop` after.
- If the branch you're pulling from has no shared history with whatever was
  previously merged into `Resources/` (e.g. switching to a different thermomatch
  branch), you'll get `fatal: refusing to merge unrelated histories`. In that case,
  don't force a merge — instead do a clean replace:
  ```bash
  git rm -rq Resources
  git commit -m "Remove Resources/ before re-importing from thermomatch/<branch>"
  git subtree add --prefix=Resources thermomatch <branch> --squash -m "Add 'Resources/' from thermomatch commit <sha> (branch <branch>)"
  ```
- **Always verify** the result before trusting it — `git subtree add --squash` has
  been observed to occasionally grab the entire thermomatch repo root instead of
  just the `Resources/` subtree (a tooling bug, not a conflict — it fails silently).
  Check it directly:
  ```bash
  git ls-tree thermomatch/<branch> -- Resources   # expected tree hash
  git ls-tree HEAD -- Resources                   # what actually landed
  ```
  If they don't match, discard the bad commit(s) and rebuild by hand from a plain
  split instead, which is more reliable than `add`/`pull` with `--squash`:
  ```bash
  git subtree split --prefix=Resources thermomatch/<branch>   # prints a commit sha
  git rm -r --cached -q Resources && rm -rf Resources
  git read-tree --prefix=Resources -u <split-sha>
  git commit -m "Add 'Resources/' from thermomatch commit <sha> (branch <branch>)"
  ```

### Pushing local edits back to thermomatch

If you edit files under `Resources/` directly in this repo, commit them normally,
then propagate upstream:

```bash
git subtree push --prefix=Resources thermomatch <branch>
```

### Switching which thermomatch branch `Resources/` tracks

There's no persistent "tracked branch" state — each `subtree pull`/`add` just names
the branch explicitly. To move to a different thermomatch branch, fetch it and pull
(or, if histories are unrelated, use the clean-replace steps above) with the new
branch name in place of `<branch>`.

### Don't gitignore it

Unlike `data-in/` raw data elsewhere in this repo, nothing under `Resources/` should
be excluded via `.gitignore` — subtree needs the full tracked content to diff against
thermomatch.
