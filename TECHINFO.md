# Technical notes

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
