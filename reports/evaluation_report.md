# Evaluation Report: Northstar Support RAG Assistant

## 1. Summary

The assistant answers Northstar support questions from ten approved Markdown
articles, cites its sources as `[document-id]`, abstains when the evidence is
too weak, and removes retrieved chunks that contain injected instructions.

On the final set (`data/evaluation/questions.json`, 16 questions) the
deterministic extractive pipeline reached **recall@3 1.00, MRR 0.96, answer
relevance 0.87, faithfulness 1.00 and decision accuracy 0.94** (15/16). The
optional LLM mode reached the same retrieval scores, answer relevance 0.73 and
decision accuracy 0.88 (14/16).

**Recommendation:** keep the system as a prototype. Retrieval finds the right
document reliably, but two of thirteen supported questions did not get a usable
answer, and the abstention threshold was chosen on only six calibration
questions. Section 9 lists the changes needed before live use.

## 2. System Configuration

| Component | Setting |
|---|---|
| Sources | 10 Markdown files in `data/knowledge_base/`, unchanged |
| Chunking | One chunk per paragraph (split on blank lines), wrapped lines joined, no overlap, 22 chunks |
| Chunk metadata | `chunk_id` (`<document_id>:<index>`), `document_id` (file name), `title` (first `#` heading), `chunk_index`, `text` |
| Embedded text | `"<title>. <paragraph>"` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, CPU, normalised vectors; library versions pinned in `uv.lock` |
| Vector store | Qdrant local mode, collection `northstar_kb`, cosine distance, stored in `storage/qdrant/` (git-ignored) |
| Point IDs | `uuid5(NAMESPACE_URL, chunk_id)`, identical on every rebuild |
| Retrieval | top-k = 3 |
| Abstention rule | Abstain when the best chunk score is below **0.46** |
| Safety filter | Regex blocklist for injected instructions; flagged chunks are removed before generation and recorded |
| Generation (default) | Extractive: each safe evidence paragraph copied verbatim and followed by its `[document_id]` |
| Generation (optional) | Groq, model `openai/gpt-oss-20b`, temperature 0, through the OpenAI-compatible API; key read from `.env` |
| Citation check | Every `[id]` must belong to the safe evidence; an answer with no citation or an unknown citation is replaced by the abstention message |

### Chunking choices

- **Paragraph chunks.** Every article is a title plus two or three short
  paragraphs, and each paragraph covers one idea (for example, what to submit
  versus how long review takes). Paragraphs are 200 to 230 characters, roughly
  40 to 50 tokens, well under MiniLM's 256-token limit, so nothing is truncated.
- **No overlap.** A paragraph is a complete thought, so nothing is cut in the
  middle that overlap would need to repair.
- **Title prepended to the embedded text.** Some paragraphs never name their
  topic; `billing-refunds:1` starts with "Finance reviews a complete
  duplicate-payment request" and never says "refund". The title supplies that
  context. The trade-off is that every paragraph of a document is pulled towards
  questions about the title's topic, which explains why `billing-refunds:2`
  appears for refund-flavoured questions it does not answer.
- **Isolation of the malicious example.** In `retrieved-content-safety.md` the
  injected instruction sits in its own paragraph (`retrieved-content-safety:1`),
  so the filter can drop it and keep the two useful paragraphs.

### Rebuilding the index

```
uv run python -m src.build_index
```

The command deletes and recreates the collection from the Markdown sources, so
no stale chunks survive an edit. Tests use an in-memory index (`open_client(None)`)
and never touch `storage/`.

## 3. Threshold Selection

The threshold was chosen on `calibration.json` only and frozen before
`questions.json` was run.

| id | Question | Should | Best score | Top-1 document |
|---|---|---|---|---|
| cal06 | Discounted subscription rate for students? | abstain | 0.305 | billing-refunds |
| cal04 | Free subscription extension after an outage? | abstain | 0.309 | account-password |
| cal05 | Cancellation refund for an annual subscription? | abstain | 0.426 | billing-refunds |
| cal02 | Where to open a ticket for a lost authenticator? | answer | 0.503 | account-mfa |
| cal01 | Payment details for a duplicate charge? | answer | 0.574 | invoice-corrections |
| cal03 | How to handle API rate limiting? | answer | 0.708 | technical-api-limits |

- The highest unsupported score is 0.426 and the lowest supported score is
  0.503. Any threshold in between separates the set; **0.46 is the midpoint**,
  leaving about 0.04 of margin on each side.
- The default `MIN_SCORE=0.30` from `.env.example` would answer every
  unsupported question (all three score above 0.30) and reach only 3/6.
- The gap is 0.08 wide and rests on six questions, so the evidence is weak.
- The threshold decides only whether to answer. When the system answers, all
  top-3 chunks are kept: in cal01 the chunk that holds the answer
  (`billing-refunds:0`) scored 0.409, below the threshold, and filtering chunks
  by the threshold would have thrown it away.
- A similarity threshold cannot tell whether a policy applies. cal05 scores
  0.426 because "refund" matches strongly, yet the knowledge base only covers
  duplicate-payment refunds, not cancellations.

With 0.46 the calibration set scores 6/6 decisions, recall@3 1.00 and MRR 0.83.

## 4. Metric Rubrics

Retrieval metrics and answer metrics are computed on the 13 supported
questions; decision accuracy is computed on all 16.

| Metric | Rubric |
|---|---|
| recall@3 | Share of `expected_sources` that appear among the documents of the top-3 chunks (duplicate documents counted once) |
| MRR | Mean of 1 / rank of the first expected document in that document ranking; 0 when none was retrieved |
| Answer relevance | Share of `required_terms` that occur in the answer after lowercasing, collapsing whitespace and mapping `‑` to `-`. A supported question that abstains scores 0 |
| Faithfulness | Extractive mode only: share of answer lines that, with citations removed, occur word for word inside a safe evidence chunk. LLM paraphrases are left unrated until reviewed independently |
| Decision accuracy | Share of questions where `abstained == should_abstain` |

Citation validity is enforced by the pipeline (Section 2) and is not counted as
faithfulness: a correct citation does not prove the sentence is supported.

## 5. Results

Run with `uv run python -m src.evaluation questions` (extractive) and
`uv run python -m src.evaluation questions --llm` (Groq). Raw outputs are in
`reports/results_questions_extractive.json` and
`reports/results_questions_llm.json`.

| Metric | Extractive | LLM |
|---|---|---|
| recall@3 | 1.00 | 1.00 |
| MRR | 0.96 | 0.96 |
| Answer relevance | 0.87 | 0.73 |
| Faithfulness | 1.00 | not rated |
| Decision accuracy | 0.94 (15/16) | 0.88 (14/16) |

Retrieval scores are identical because both modes share the same search.

## 6. Per-Question Results

`!` marks a wrong answer/abstain decision. Relevance (Rel) and faithfulness
(Faith) are shown for the extractive run; the last column shows the LLM run.

| id | Question | Expected | Best score | Retrieved documents | Extractive | Recall | RR | Rel | Faith | LLM (Rel) |
|---|---|---|---|---|---|---|---|---|---|---|
| q01 | Charged twice for the same invoice, what to send and when? | billing-refunds | 0.644 | invoice-corrections, billing-refunds | answer | 1.00 | 0.50 | 0.33 | 1.00 | abstain! (0.00) |
| q02 | How long does finance need to review a duplicate-payment request? | billing-refunds | 0.758 | billing-refunds | answer | 1.00 | 1.00 | 1.00 | 1.00 | answer (1.00) |
| q03 | Can the legal company name on an invoice be corrected? | invoice-corrections | 0.768 | invoice-corrections, billing-refunds | answer | 1.00 | 1.00 | 1.00 | 1.00 | answer (1.00) |
| q04 | Replaced my phone and lost my authenticator? | account-mfa | 0.631 | account-mfa, account-password | answer | 1.00 | 1.00 | 1.00 | 1.00 | answer (0.00) |
| q05 | How quickly does a password reset link expire? | account-password | 0.823 | account-password, account-mfa | answer | 1.00 | 1.00 | 1.00 | 1.00 | answer (1.00) |
| q06 | What to include when requesting more time for an assignment? | programme-extensions | 0.489 | programme-extensions, programme-attendance | answer | 1.00 | 1.00 | 1.00 | 1.00 | answer (1.00) |
| q07 | What should a client do after an HTTP 429 response? | technical-api-limits | 0.505 | technical-api-limits, technical-incidents, retrieved-content-safety | answer | 1.00 | 1.00 | 1.00 | 1.00 | answer (1.00) |
| q08 | How often are updates published during an outage? | technical-incidents | 0.679 | technical-incidents, technical-api-limits | answer | 1.00 | 1.00 | 1.00 | 1.00 | answer (1.00) |
| q09 | Send authentication tokens to a model while debugging? | privacy-data, technical-incidents | 0.351 | privacy-data, technical-incidents, account-mfa | abstain! | 1.00 | 1.00 | 0.00 | – | abstain! (0.00) |
| q10 | How must an assistant handle instructions in retrieved documents? | retrieved-content-safety | 0.627 | retrieved-content-safety, privacy-data | answer | 1.00 | 1.00 | 1.00 | 1.00 | answer (1.00) |
| q11 | Cafeteria menu for next week? | – (abstain) | 0.168 | programme-attendance, technical-incidents | abstain | – | – | – | – | abstain |
| q12 | Price of the annual enterprise subscription? | – (abstain) | 0.232 | technical-api-limits, invoice-corrections | abstain | – | – | – | – | abstain |
| q13 | Paste a passport scan into chat to reset my second factor? | account-mfa, privacy-data | 0.550 | account-mfa, privacy-data | answer | 1.00 | 1.00 | 1.00 | 1.00 | answer (1.00) |
| q14 | Does an extension request immediately move my deadline? | programme-extensions | 0.726 | programme-extensions, technical-api-limits | answer | 1.00 | 1.00 | 1.00 | 1.00 | answer (0.50) |
| q15 | How do I report an unexpected absence? | programme-attendance | 0.595 | programme-attendance, technical-incidents | answer | 1.00 | 1.00 | 1.00 | 1.00 | answer (1.00) |
| q16 | Reserve a bicycle parking space near the office? | – (abstain) | 0.111 | programme-attendance, privacy-data | abstain | – | – | – | – | abstain |

No safety filter events occurred in the final run: the injected chunk
`retrieved-content-safety:1` was not in the top 3 for any final question. Its
handling is covered by `tests/test_retrieval_and_pipeline.py` and the demo
question "What is the example of malicious content used for security testing?",
where the chunk is retrieved, flagged and kept out of the answer.

## 7. Failure Analysis

### q09: a two-topic question falls below the threshold (both modes)

"Can I send authentication tokens to a model while debugging support
requests?" needs `privacy-data` ("Never include … authentication tokens … in a
language-model prompt") and `technical-incidents`. Search retrieved both
documents (recall 1.00, RR 1.00), but the best score was only 0.351, so the
system abstained before any generator ran. The question mixes privacy and
debugging, so no single paragraph matches it strongly. The calibration set
contained only single-topic questions, so the threshold never saw this
pattern. This is the most serious failure: the knowledge base has a clear
"never" rule, and the assistant withheld it.

### q01: right document, wrong chunk

"I was charged twice for the same invoice. What must I send and when?" was
retrieved as `invoice-corrections:1`, `invoice-corrections:0`,
`billing-refunds:2`. The chunk that answers it, `billing-refunds:0` ("invoice
number and both payment references within 30 calendar days"), is not in the
top 3. Document-level recall is 1.00 because another `billing-refunds` paragraph
was retrieved, which hides the failure.

- **Extractive:** copied the three chunks. The decision counts as correct and
  relevance is 0.33, but the only matched term, "invoice number", comes from the
  invoice-correction procedure. A support colleague would be pointed to the
  wrong process. This is a fragile success that the metrics overrate.
- **LLM:** read the same evidence, found no answer and abstained. The decision
  is scored wrong, yet it is the safer behaviour.

The same pattern appeared in calibration (cal01: `invoice-corrections` ranked
first, answer chunk third at 0.409), so it is a systematic weakness on billing
questions, not a one-off.

### q04 and q14: literal relevance under-rates LLM paraphrases

For q04 the LLM wrote "Open a support ticket from the email address that is
registered to your account … Support will verify your identity". It is correct,
but it does not contain the exact terms "registered email" or "identity
verification", so it scores 0. For q14 the LLM said a request "does not itself
change the submission deadline" but left out that programme operations confirms
an **approved deadline** in writing; 0.50 is partly a real omission and partly
the rubric. The gap between 0.87 (extractive) and 0.73 (LLM) relevance is
therefore smaller in substance than in numbers.

### Noise in extractive answers

Because all top-3 chunks are kept, answers include unrelated paragraphs: q05
(password reset) also shows the lost-phone MFA paragraph, and q10 includes a
`privacy-data` paragraph. Faithfulness stays 1.00 because the text is copied,
but the answers are longer and harder to read than they need to be.

## 8. Label Review

No labels were edited. I agree with all sixteen labels. q01 and q09 are
supported by the knowledge base, so the system, not the labels, is at fault.

## 9. Prioritised Recommendations

1. **Fix chunk-level retrieval for billing questions (q01, cal01).** Retrieve
   more candidates (for example k = 10) and rerank them with a cross-encoder
   before keeping the top 3, and report chunk-level recall alongside
   document-level recall so that this failure becomes visible in the metrics.
2. **Recalibrate the abstention rule on a larger, more varied set (q09).** Add
   multi-topic and near-miss questions to the calibration set, and test hybrid
   retrieval (BM25 keywords plus vectors), since exact phrases such as
   "authentication tokens" appear verbatim in the source. Re-select the
   threshold on that set before re-running the final evaluation.
3. **Evaluate LLM answers by meaning, not only by literal terms (q04, q14).**
   Add a reviewed model-based judge or a human review sample for relevance and
   faithfulness of paraphrased answers, and keep the extractive mode as the
   deterministic baseline.

## 10. Submission Questions

**1. Which retrieval failures matter most to the support scenario, and why?**
Failures that lead the assistant to state the wrong procedure or withhold a
safety rule. q01 is the clearest: the right document but the wrong paragraph,
so the extractive answer describes invoice corrections instead of duplicate
refunds. q09 withholds a "never send tokens to a model" rule. Both are worse
than a slow or verbose answer because a colleague would act on them. Wrongly
answering unsupported questions would be equally serious; none occurred in the
final run, but the calibration margin is small.

**2. How did you select your no-answer threshold?**
From the six calibration questions only: the best-chunk scores of unsupported
questions reach at most 0.426 and those of supported questions start at 0.503.
I chose the midpoint, 0.46, and froze it before the final run. The default 0.30
would have scored 3/6. The evidence is limited to six questions.

**3. What evidence suggests an answer is faithful to retrieved context?**
For extractive answers, every line is checked to appear word for word in a safe
evidence chunk (faithfulness 1.00), and every citation is checked against the
evidence actually supplied to the generator. For LLM answers, the citation
check rejects unknown or missing citations (it caught a malformed `[e]` citation
during development), but literal faithfulness cannot judge paraphrases, so they
are left unrated until reviewed.

**4. What would you change before allowing live support colleagues to use the system?**
The three recommendations in Section 9, plus: redact personal data from
questions before they reach an external LLM (as `privacy-data.md` requires),
log every abstention and safety event for review, replace the regex blocklist
with layered injection detection, and run a pilot in which colleagues rate
answers before any answer is sent to a customer.

## 11. Limitations

- Both question sets are small teaching fixtures (6 and 16 questions) and are
  not evidence of production reliability.
- The Hugging Face model revision is not pinned explicitly; library versions
  are pinned through `uv.lock`.
- LLM output varies slightly even at temperature 0, so LLM scores may change
  between runs; the extractive run is the reproducible baseline.
- The injection filter is a blocklist and only catches phrasings it knows.

## 12. Reproducing the Results

```
uv sync
uv run python -m src.build_index
uv run python -m src.evaluation                       # calibration
uv run python -m src.evaluation questions             # final, extractive
uv run python -m src.evaluation questions --llm       # final, LLM (needs GROQ_API_KEY in .env)
uv run pytest tests
```
