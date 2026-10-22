# Independent assembled-package verification

Reviewer: fresh verifier, separate from source reviewers and coordinator. Reviewed 22 October 2026. Final prose recheck used `/private/tmp/deepsilencer-staging3/deepsilencer-paper`; final crop/supplement/status recheck used `dist/DeepSilencer-agent/skill/deepsilencer-paper`.

## Coverage and method

Read the assembled SKILL.md, complete navigation index, main-paper prose and all captions, four CSV main tables, and supplement pointer document. Visually surveyed all 24 original PDF pages through six contact sheets. Opened full original previews for pages 4, 6, 21, 22 and 24 to inspect small table symbols, equations, hats and wrapped DOI strings. Opened every delivered main figure (1–7), all four equation images, and Table 5 at image resolution. Inspected their panel labels, axes, legends, table headings and small scientific notation. Compared Table 1's 24 feature names/descriptions/contributions against the visible original page 4. Compared Tables 2–4 against visible source tables and prose context. The chosen figures retain their complete panels and scientific labels; Figure 4's AUROC/AUPRC legends, Figure 5's panel letters/dimension axes, and Figure 7's 19 positions and A/C/G/U rows are readable. Table 5 retains both groups and all eight example rows.

Read aggregate/document verification reports and the complete review queue. Checked the three delivered associated CSVs byte-for-byte against the extracted source CSVs: Hu 2361, Mix 472, Taka 702 data rows; all share the retained `siRNA,mRNA,label,y,td` columns. This checks transfer fidelity, not biological validity of every data cell.

## Findings and repair recheck

1. Original assembly joined the Regression Task and Contrastive Task into one numbered paragraph. Original page 6 visibly has separate items 2 and 3. Coordinator repaired this; staging3 contains separate list items.
2. Original assembly interrupted the sentence beginning "superior performance on" with Figure 3 and Table 2. Original pages 10–11 confirm the continuation "the Mixset and Takayuki datasets". Coordinator repaired semantic reading order; staging3 has the complete paragraph before the floats.
3. Original markup fragmented β=2.4 and the sampling decimal values, and placed prediction hats awkwardly. Against original page 6, staging3 now retains β=2.4 and visible hat notation; the displayed equations remain exact source images. Staging2/3 sampling prose preserves the source easy (0.2,0.4), hard (0.05,0.1), and default (0.05,0.2) values. Algorithm indices are explicit `_1`/`_2`, preserving the source mathematics and condition `α_1 < |y_1 − y_2| < α_2`.

No remaining substantive omission, false sentence join, truncated figure panel, or changed scientific value was identified in this review. The source's statements, including imperfect scientific terminology or internally surprising examples, were retained rather than silently corrected.

## Exact diagnostic adjudications supported by visible originals

The six remaining paired numerical checks are parser-only line-wrap differences, not missing paper numbers. Both `number_differences` and `independent_parser_number_differences` are supported for each page/fingerprint below:

- Page 21, fingerprint `360b6c531eb77cee0b7f8cf6a46f8bf53ae0a769b085f5bffa603d9c30047214`: original reference 2 DOI breaks visibly after `10.` before `1038/s41573-024-00912-9`. Assembled `10.1038/s41573-024-00912-9` faithfully joins the URL.
- Page 22, fingerprint `ac416289df4a6528e8351bc0bd2d23f5abae57317ea9bdb9b684ea6190c52546`: original references 13, 17 and 19 wrap between `10.` and `18653`, `1101` and `1093`; reference 16 wraps between `1989.1.4.` and `541`. Assembled DOI strings preserve the visible digits and punctuation in correct order.
- Page 24, fingerprint `a965d3cfec087ef2b4424c17d01b6fe1fb1df510243b32fba6fbeb9dbcc74f96`: original reference 36 wraps after `10.` before `32604/cmc.2022.022901`. The assembled joined DOI is faithful.

Final report recheck confirms aggregate status `reviewed_with_limitations`, `mechanical_ok: true`, all pages reviewed, six applied adjudications and zero stale adjudications. Final Smooth L1 and Figure 1 crops were reopened: adjacent text ink is removed, the full equation and flowchart remain intact. Final supplementary captions explicitly identify repository CSV provenance, correct row counts/schema and the td feature-vector units/order limitation. No additional defect was identified. The coordinator separately reported strict verification exit 0; this reviewer inspected the resulting JSON status.

## Retrieval test using only the assembled package

Question: What label-difference interval does default sampling use, what total loss is reported, and does adding thermodynamic features help cross-dataset performance?

Answer retrieved from the package: Section 2.2 Algorithm 1 and Section 3.3 sampling ablation specify strict bounds `0.05 < |y_1 − y_2| < 0.2`. Section 2.3's total-loss image gives `L_total = L_cls + L_reg + 2 × L_cont`; classification loss uses siRNA1, while regression uses both sequences. Section 3.2/Table 3 reports sequence-only DeepSilencer AUC/F1 of 0.802/0.760, increasing to 0.820/0.775 with TR. The package exposes each answer through searchable text and the linked equation/table without the original PDF.

## Limitations

This is fidelity/retrieval review, not replication of model training or certification of paper claims. Most running prose was compared through contact-sheet survey plus assembled-text reading rather than a second independent character-by-character transcription of all 24 pages. Tiny figure labels were inspected in delivered crops; source resolution limits remain. Table 5 remains an image and cannot be queried as CSV. Associated CSV files lack a formal thermodynamic units/order dictionary, as the conversion notes correctly state. Their 3535 combined supplied rows should not be assumed to equal the paper's reported 3714-study aggregate. No separate supplementary PDF was present in this reviewed bundle.
