# Hand check of 200 labels (Gate 1)

Labels come from string matching (D6). This check measures how often that matching gives the wrong label. **Gate 1 passes if fewer than 10% of the 200 labels change after the hand check (fewer than 20 of 200).** Probe results are then reported with and without the flagged questions, as the proposal requires.

## Steps

1. Once the full labels exist, draw the sheet from the training and test files:
   ```bash
   uv run python -m saerag.eval.handcheck sample --files nq_train triviaqa_train nq_test triviaqa_test popqa_test hotpotqa_test
   ```
   This writes `runs/handcheck/sheet.csv`: a simple random sample of 200 questions with seed 13. Random sampling keeps the error rate an unbiased estimate. Do not swap rows out by hand.
2. Upload it to the shared drive as a Google Sheet. Split the rows between two checkers if you like. Do 20 rows together first to align on the rules below.
3. Fill in `verdict_closed`, `verdict_bm25` and `verdict_gold` using the rules below.
4. Download the sheet as CSV over `runs/handcheck/sheet.csv` and run:
   ```bash
   uv run python -m saerag.eval.handcheck score
   ```
   This recomputes every label with your verdicts through the same label rule and prints the label error rate and PASS/FAIL. It writes the flagged qids to `runs/handcheck/flagged.json`.

## Columns

| Column | Meaning |
|--------|---------|
| `question`, `aliases` | The question and every accepted answer, separated by ` \| ` |
| `{closed,bm25,gold}_output` | The model's reply in each run. Blank means that run does not exist (no gold passage) |
| `{closed,bm25,gold}_auto` | What string matching decided |
| `verdict_{closed,bm25,gold}` | **You fill this in.** `y` means the reply is correct, `n` means it is wrong, and blank means you agree with `auto` |
| `short_alias` | TRUE when an alias has 3 characters or fewer. Check these rows with extra care |
| `label`, `abstain_reason` | The automatic label, for reference only. Do not edit it |
| `notes` | Anything odd: a bad gold answer, an ambiguous question, a date problem |

## What counts as correct

Judge whether a reader who knows the answer would accept the reply. Ignore how it was matched.

- **Correct:** the reply names the answer entity, even with extra words ("It was Paris, France"). Spelling variants and transliterations count, and so do full names when the alias is a surname. A number in a different format also counts ("1,000" and "1000", "3" and "three").
- **Wrong:** the reply names a different entity, even if an alias appears inside it by accident. For example, "AR-15" for the alias "AR", "Paris Hilton" for "Paris", or "PL/SQL" for "PL". It is also wrong if the reply lists several candidates and hedges between them ("Either Lyon or Paris"), or if it contains the alias only while repeating the question.
- **Gold answer is wrong or outdated:** judge against the dataset's answer anyway, and write `bad gold` in `notes`. The labels are defined by the dataset, and we report these cases separately.
- **Unsure:** leave the verdict blank and write `unsure` in `notes`. Two checkers then settle these together before scoring.

## Things to watch (D6)

- **Short PopQA aliases** such as "AR", "PL" and "US" match any reply that contains that token.
- **Token-span matching** stops "1" from matching "1990", but it still matches "1990" in "between 1990 and 1995".
- **HotpotQA yes/no questions:** "yes" matches "Yes, but ...".
- **Self-Aware** rows are ABSTAIN by construction. Only check that the question really is unanswerable.

## What to report

- The label error rate with its 95% interval, which is about ±4 points at n = 200.
- The match error rate per run.
- Error counts by dataset and for `short_alias` rows.
- The list of flagged qids. Every main probe table is repeated without them.
