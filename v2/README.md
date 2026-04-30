# Relative clause extractor — v2 (post–PI validation)

This directory is the **recommended release** ([`v2/`](https://github.com/alarakaymak/relative-clause-extractor/tree/main/v2) in the public repo): extraction code and frozen cohort outputs revised after external validation (~early 2026). The **root-level** `relative_clause_extractor.py` on `main` predates those changes; use **`v2/`** for current behavior and for the bundled results.

## Problems flagged in validation (baseline)

Roughly speaking, reviewers reported:

- **Over-extraction**: Some complement or content clauses (~“that S” with **its own subject**) looked like restrictive RCs → SRC false positives (~10%+ concern in sampled rows).
- **SRC vs ORC**: Object relatives with **an intervening subject** inside the RC (e.g. “for **which she** is an honoree”, “that mostly **everyone** had”) were labelled **SRC** when they should behave like **gap/object** relatives (**ORC**).
- **`whose`**: Poor coverage or everything lumped into **Other** instead of distinguishing SRC-style possessed subjects vs others.
- **`where` / `when` / `why`**: Over-use of **SRC/ORC** when the clause behaves like an **adjunct** (better as **Other**).
- **Data hygiene**: Repeated rows for the same `(sentence, relative_clause)`, and bogus **`head_noun`** values (punctuation-only or non-alphabetic starts).

Baseline exports (pre-fix) reflected these issues numerically—for example bundled **duplicate keys** (~1.9k rows across f23+f24+f25 in one older package) versus **fewer than 10** duplicate-key rows on the frozen v2 corpus run.

## What changed in code (high level)

| Area | Change |
|------|--------|
| Complements | `_is_complement_clause_by_structure` (and backups) reduces “two subjects” complements tagged as SRC. |
| Intervening subject / ORC | Word-order-aware `has_intervening_subject_by_wordorder` (with fallback when the relativizer span is brittle). |
| `whose` | Classify SRC vs ORC vs **Other** from possessed NP role + following material—not “always Other.” |
| Adjunct relativizers | `where`/`when`/`why` forced to **Other** only when dependency role is adjunct-like (`advmod`, `obl`, `npmod`, `mark`, `dep`, or missing). |
| Output quality | Skip rows where `head_noun` is empty or does not start with a letter; **deduplicate** on `(sent, relative_clause)` before writing CSV. |

Full logic lives in `relative_clause_extractor.py` in this directory.

## Bundled frozen results (`results/`)

These CSVs are the **canonical v2 totals** matching this script revision (January 2026 full run):

| Folder | Rows | SRC | ORC | Other |
|--------|-----:|-----:|-----:|------:|
| `results/f23/` | 20,664 | 15,072 | 4,636 | 956 |
| `results/f24/` | 25,472 | 18,764 | 5,358 | 1,350 |
| `results/f25/` | 2,924 | 2,097 | 669 | 158 |
| **Total** | **49,060** | **35,933** | **10,663** | **2,464** |

File per cohort: `results/<cohort>/results_cursor.csv`.

Each row includes the usual columns (`head_noun`, `relative_clause`, `rc_type`, `relativizer`, `file`, `sent`, counts, …).

## Running the extractor (regenerate outputs)

Requirements match the upstream project (**SuPar**, **NLTK**, **torch**, SuPar **`models/`** next to where you execute Python).

1. Obtain **`models/ptb.biaffine.dep.lstm.char`** and **`models/ptb.crf.con.lstm.char`** as in the [original README / Drive link](https://github.com/alarakaymak/relative-clause-extractor)).
2. Work from the **same repository root layout** used before: **`models/`** on the cwd path the script resolves (currently `models/…` relative to the process cwd).
3. Put corpus `.txt` files under **`input_texts/`** next to **`models/`** at the **repository root** (`v2/` is a sibling folder to both).

SuPar loads **`models/…` relative to the process working directory**, not relative to the script file.

4. **Recommended:** from the **repository root**, run:

   ```bash
   PYTHONPATH=v2 python v2/main.py
   ```

   Then **`result/results_cursor.csv`** is written next to **`v2/`** (root-level `result/`). Edit **`v2/main.py`** if you want outputs under **`v2/result/`** instead.

5. **Alternative:** `cd v2`, then symlink dependencies once (Unix/macOS): **`ln -sf ../models models`** (and **`ln -sf ../input_texts input_texts`** if inputs live at repo root). Run **`python main.py`** from **`v2/`**.

To match the bundled cohort layout, split or move CSVs manually into **`v2/results/f23/`**, etc.

**Note:** This folder does **not** ship **`models/`** or **`input_texts/`**; obtain them as described in the repository root [README](https://github.com/alarakaymak/relative-clause-extractor/blob/main/README.md).

To compare against an older extraction run, diff on shared keys (e.g. `sent` + `relative_clause`) against `results/<cohort>/results_cursor.csv`.
