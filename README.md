# DeepSilencer paper agent

An agent-readable research package for **DeepSilencer: A Novel Deep Learning Model for Predicting siRNA Knockdown Efficiency** by Wangdan Liao and Weidong Wang, arXiv:2503.04200v1 (6 March 2025).

**Status: partial Paper2Agent conversion.** The paper-reading skill and associated datasets are supplied. No DeepSilencer MCP server, prediction model, trained weights, training implementation, or reproduced performance is supplied. The authors' repository contains only a README and three datasets; its README says the code will be uploaded. A wrapper cannot faithfully execute an implementation that is absent.

- Paper: https://arxiv.org/html/2503.04200v1
- Authors' repository: https://github.com/BlackCattt9/DeepSilencer
- Inspected source commit: `0e43907220e61e75c414ac3f9fcab4ee80c9ae82`
- Conversion workflow: the supplied local Paper2Agent skill, including Paper2Skill and Paper2MCP source selection.

## Use the paper skill

Open [the reading skill](dist/DeepSilencer-agent/skill/deepsilencer-paper/SKILL.md) in your agent, or copy the `deepsilencer-paper` directory into your agent's supported skills directory. Start with its references index, then retrieve relevant sections and linked figures or tables. The package is portable and does not need a Python environment to read.

The linked CSV files contain the original repository datasets, exported without scientific recalculation. They are research inputs, not prediction outputs. Do not infer undocumented meanings or units for `label`, `y`, or `td` from their column names.

The supplied datasets contain 3,535 records in total (Hu 2,361; Mix 472; Taka 702). This differs from the paper's reported aggregate of 3,714 siRNAs; the files are preserved without inventing missing records or reconciling that discrepancy. Some equations and Table 5 are images; consult their linked crops for exact notation and cells.

## Verify the delivery

Run from this repository using Python 3:

```sh
python3 scripts/verify_delivery.py
```

This checks the delivered file hashes, local Markdown links, expected package layout, complete CSV structure, and the ZIP contents. It does not rerun the external PDF review or establish scientific/model correctness. The supplied verification records describe the actual source review and conversion checks.

## Missing model implementation

See [MCP status](verification/mcp-blocked.md) and [independent source scan](verification/source-scan.json). Prediction, training, model evaluation, clean-runtime installation, MCP transport calls, and a tested MCP ZIP remain deferred. OligoFormer is a dataset provenance source and was not substituted as the DeepSilencer model.

To resume MCP conversion, supply the authors' runnable code, required model artifacts, dependency instructions, and a traceable reference invocation. The workflow then requires direct source execution, minimal wrappers, fresh independent verification, installation in a clean runtime, and tests of the extracted MCP delivery.

## Attribution and rights

Paper and datasets retain their original authorship and applicable terms. The inspected upstream repository has no license file; this repository does not grant a new license over the paper or datasets. Consult the authors for reuse terms. This is a conversion of the cited materials, not an official author release or a newly implemented DeepSilencer model.
