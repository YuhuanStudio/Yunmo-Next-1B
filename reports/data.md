# Pretraining Data for Yunmo-Next-1B

*Technical report · Yunmo-Next-1B · October 2026*

## Summary

Yunmo-Next-1B saw 19.76B tokens drawn from web text, educational text, Wikipedia and Wikisource, public
government documents, mathematics and source code. Chinese text is predominantly Traditional: Simplified
sources were converted with OpenCC, protecting code and other machine text, and a small share of original
Simplified text was kept on purpose. The training data is not redistributed. This report lists every source,
its recorded license, how much of it the model saw, how the Chinese text was processed, and what the
filtering does and does not guarantee.

## 1. Sources

| Source | Content | Recorded license |
|---|---|---|
| FineWeb2-HQ (Chinese, Traditional surface) | web | ODC-By 1.0; Common Crawl terms |
| FineWeb2 Traditional Chinese, quality-filtered subset | web | ODC-By 1.0; Common Crawl terms |
| Ultra-FineWeb-zh (converted) | web | Apache-2.0 |
| OpenCSG FineWeb-Edu-zh (converted) | educational web | Apache-2.0 + OpenCSG Community License (commercial use requires OpenCSG approval) |
| IndustryCorpus2, subject and math subsets (converted) | educational / math | Apache-2.0 |
| Chinese Wikipedia, zh-tw ([own build](https://huggingface.co/datasets/yuhuanstudio/wikipedia-pretrain-zh-tw)) | encyclopedia | dataset Apache-2.0; underlying text CC BY-SA 4.0 / GFDL |
| Chinese Wikisource, zh-tw | classical and historical texts | CC BY-SA 3.0 |
| Taiwan court judgments, statutes, regulations, patents | official text | Open Government Data License, Taiwan v1 |
| FineWeb-Edu (English) | educational web | ODC-By 1.0; Common Crawl terms |
| Ultra-FineWeb L3 (English) | synthetic rewrites of web text | Apache-2.0 |
| FineMath (4+, large) | mathematics | ODC-By 1.0; Common Crawl terms |
| The Stack: source files, documentation and configuration | code | not recorded in the audit table |
| Kaggle notebooks | code | not recorded in the audit table |
| MiniMind pretraining corpus (selected, converted) and its original Simplified twins | general, math, code | mixed upstream licenses; not cleared for redistribution |

Licenses are as recorded by the project's source audit; they are not a legal review. Two entries matter for
anyone using the weights commercially: the OpenCSG Community License requires OpenCSG approval for
commercial use of that dataset, and the MiniMind material has mixed upstream licenses. The weights are
released under Apache-2.0 and no training text is distributed, but whether those dataset terms reach a model
trained on them is a legal question this report does not answer.

## 2. How much of each source the model saw

Tokens consumed by source family through the end of the 14.4B-token capability stage:

| Family | Tokens | Origin |
|---|---:|---|
| FineWeb2-HQ (Traditional surface) | 5.108B | Traditional script; native vs converted origin not resolved |
| Ultra-FineWeb L3 English (synthetic rewrites) | 2.318B | English, synthetic |
| The Stack, executable code | 1.177B | code |
| FineWeb-Edu English | 0.994B | English |
| Wikisource (classical) | 0.912B | Traditional, classical |
| FineMath large | 0.831B | English math |
| Ultra-FineWeb-zh | 0.768B | converted Traditional |
| Chinese Wikipedia | 0.576B | converted Traditional |
| IndustryCorpus2 subjects | 0.384B | converted Traditional |
| FineMath 4+ | 0.332B | English math |
| OpenCSG FineWeb-Edu-zh | 0.288B | converted Traditional |
| IndustryCorpus2 math | 0.253B | converted Traditional |
| The Stack, documentation and config | 0.210B | code |
| Government documents | 0.105B | native Traditional |
| Kaggle notebooks | 0.101B | code |
| FineWeb2 Traditional subset | 0.042B | native Traditional |

The largest source, FineWeb2-HQ, is Traditional on the surface, but whether each document was written in
Traditional Chinese or converted upstream is not resolved, so the report does not state a "native
Traditional" share.

The final 5.36B tokens (extension and anneal) came from one rebuilt 16K pack, described in §4.

### 2.1 Domain mix by stage

Share of tokens by content domain:

| Domain | K24–K48 (8K) and K60 (16K) | K48 32K | K60 capability |
|---|---:|---:|---:|
| General web | 40.2% | 64.2% | 40.6% |
| General knowledge / education | 24.9% | 3.0% | 18.0% |
| Code and computer science | 10.4% | 9.0% | 10.2% |
| Mathematics and physical science | 10.3% | 3.4% | 9.0% |
| Books and humanities | 5.5% | 18.6% | 8.0% |
| Education | 4.1% | 1.0% | 8.0% |
| Encyclopedia | 4.1% | 0.8% | 4.0% |
| Law, economics, society | 0.5% | 0.0% | 2.2% |

The 32K stage shifted toward documents long enough to fill its window; the capability stage raised
education, books and law relative to the base mixture.

## 3. Traditional Chinese processing

- **Conversion.** Simplified Chinese prose is converted with OpenCC `s2twp`, which converts characters and
  also maps vocabulary to the Taiwan standard (e.g. 軟件 → 軟體).
- **Protected spans.** Inline code, URLs, HTML code/pre blocks and tags, selected math delimiters and fenced
  code blocks are left untouched; an unclosed fence protects the rest of the document and is flagged. The
  converter tracks four modes per span: converted prose, original Simplified, code, other language.
- **Kept Simplified text.** A parallel set of original Simplified documents (21.3M tokens, the exact
  pre-conversion twins of already selected material) is kept unconverted for reading. It is a second script
  view of the same content, not additional knowledge.
- **Re-deduplication.** After conversion the selected Chinese web text was re-tokenized and deduplicated
  again: of 987,130 documents, 970,545 changed under conversion, 765 became exact duplicates of another
  document and were removed, 757 matched known exclusions, and 117 had unclosed code fences.
- **Unfiltered Chinese Wikipedia** was not used directly, partly because its Traditional text mixes regional
  variants; the project's own zh-tw build was used instead. There is no corpus-wide regional-vocabulary
  filter beyond what `s2twp` does.

## 4. The extension and anneal pack

| Property | Value |
|---|---|
| Rows × context | 206,155 × 16,384 |
| Input tokens | 3,377,643,520 |
| Supervised labels | 3,373,795,147 (3.85M cross-document labels masked) |
| Language (document-level identification) | 58.0% Chinese, 31.3% English, 9.7% code, 0.7% unmeasured |

| Component | Rows | Share |
|---|---:|---:|
| Chinese web (Traditional) | 83,346 | 40.4% |
| English | 46,908 | 22.8% |
| Source code | 16,234 | 7.9% |
| Classical books | 13,121 | 6.4% |
| MiniMind general | 10,069 | 4.9% |
| Wikipedia (zh-tw) | 8,655 | 4.2% |
| Mathematics (two lanes) | 15,860 | 7.7% |
| Structured code | 2,267 | 1.1% |
| Kaggle notebooks | 1,403 | 0.7% |
| MiniMind code / math / English-mixed | 5,478 | 2.7% |
| Original Simplified twins | 1,289 | 0.6% |
| Government documents | 1,525 | 0.7% |

Document lengths in this pack: 5.7% of tokens are in documents under 256 tokens, 38.7% in 513–2,048,
34.5% in 2,049–8,192 and 10.6% in 8,193–16,384.

**Repetition.** The extension consumed 4.161B tokens from this 3.378B-token pack; in its seed-17 shuffled
order, 47,817 of 253,972 sampled rows were second visits. The anneal drew another 1.2B tokens from the same
pack before the released checkpoint, for about 1.59 passes in total.

## 5. Hygiene and its limits

- **Deduplication** by exact content hash, repeated after Traditional conversion.
- **Decontamination** by normalized substring matching against evaluation inputs (90,400 benchmark needles
  for the extension pack) and exclusion of the project's held-out documents.
- **Known-bad exclusions**: 1,064 documents flagged by manual review or rules (broken, templated, or
  otherwise defective) were removed from the extension pack; 41 documents were read by hand during that
  review.

What this does not guarantee:

- No near-duplicate or semantic decontamination across the full corpus; paraphrased benchmark content can
  remain.
- No factual-correctness audit of web, educational, synthetic or encyclopedic text.
- Domain and language shares are measured by source labels and document-level language identification, not
  per-token truth.
- No ablation isolates the effect of Traditional conversion, protected spans, filtering or the source mix.
- There is no complete per-document exposure census across all stages.
