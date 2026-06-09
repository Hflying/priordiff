# PriorDiff: Transferable Structural Priors in Continuous Diffusion Language Models

Reference code for the EMNLP submission
**"Transferable Structural Priors in Continuous Diffusion Language Models."**

This repository contains the training, decoding, and evaluation pipeline behind:

- the **V_i** position-template constrained-decoding rule and its instance for
  continuous diffusion (LCVR);
- the **M3-v1** MLM-based refinement of LCVR outputs;
- the **boundary-aware mixed training** (BMT / M1) recipe that suppresses
  boundary attention drift (BAD);
- the **cross-domain backbone transfer** experiments (ci ↔ sonnet ↔ E2E);
- the AR-LM + constrained-decoding (AR+CD) baseline used in the wallclock and
  lexical-ceiling analyses.

Built on top of the Diffusion-LM family of codebases.

---

## Repository layout

```
improved-diffusion/
  improved_diffusion/       core diffusion package (model, training, rounding,
                            allusion / structural conditioning)
  scripts/                  training entry points + decoding (LCVR), AR+CD
                            baselines, wallclock benchmark, MLM refinement,
                            diagnosis tools (BAD / VPD), evaluation scripts
  setup.py                  installable package
bash/                       reproducible orchestration scripts for the main
                            ci / sonnet / E2E experiments and transfer studies
train_run.py                top-level launcher
training_args.json          example training configuration
```

`datasets/`, `diffusion_models/`, `classifier_models/`, `ar_baseline/`, the
HuggingFace `transformers` fork, generation outputs, training logs, and the
LaTeX paper sources are **not** included; they are either too large, should
be obtained / produced locally, or kept out of the public code repo. See
*Reproducing the paper* below.

---

## Environment

Tested with Python 3.9, PyTorch 1.13+, CUDA 11.x.

```bash
# core diffusion package
pip install -e improved-diffusion

# HuggingFace transformers (fork required by the AR baseline + classifier
# guidance; see paper appendix for the small patch to run_clm.py / trainer.py)
pip install "transformers>=4.20,<4.30" datasets accelerate
pip install nltk sacrebleu sentencepiece spacy blobfile mpi4py wandb tqdm
```

---

## Reproducing the paper

The `bash/` directory groups runnable recipes by experiment. The most
relevant ones for the EMNLP paper are:

| Experiment            | Script(s)                                         |
|-----------------------|---------------------------------------------------|
| ci diffusion train    | `bash/ci_diffusion_sz12.sh`                       |
| BMT (M1) training     | `bash/ci_diffusion_bmt.sh`                        |
| ci LCVR + M3 decoding | `improved-diffusion/scripts/decode_pkl_m3_v1.py`  |
| Sonnet pipeline       | `bash/sonnet_train.sh`, `decode_sonnet_m3_v1.py`  |
| E2E pipeline          | `bash/e2e_train.sh`, `decode_e2e_m3_v1.py`        |
| AR+CD baseline (ci)   | `improved-diffusion/scripts/ar_ci_cd_sample.py`   |
| AR+CD baseline (son.) | `improved-diffusion/scripts/ar_sonnet_cd_sample.py`|
| Cross-domain transfer | `bash/ci_finetune_from_sonnet.sh`, `bash/ci_finetune_from_e2e.sh` |
| Wallclock benchmark   | `improved-diffusion/scripts/bench_wallclock.py`   |
| BAD diagnosis         | `improved-diffusion/scripts/diagnose_attention.py`|
| VPD diagnosis         | `improved-diffusion/scripts/diagnose_vocab_projection.py` |
| Token-level vs latent | `improved-diffusion/scripts/eval_token_level_posthoc_baseline.py` |
| Rejection-sampling RS | `improved-diffusion/scripts/eval_rejection_sampling_baseline.py` |

Datasets:

- **ci**: full-Tang Song lyric corpus (80k lines); pre-processing scripts are
  in `improved-diffusion/scripts/` (search for `ci`).
- **sonnet**: 2.7k Shakespearean sonnet lines.
- **E2E**: standard E2E-NLG cleaned corpus.

These are obtained from their original sources; the paper appendix lists the
exact URLs and preprocessing steps.

---

## Citation

```bibtex
@inproceedings{priordiff2026,
  title  = {Transferable Structural Priors in Continuous Diffusion Language Models},
  author = {Anonymous},
  booktitle = {Under review at EMNLP 2026},
  year   = {2026}
}
```

---

## Acknowledgements

This codebase builds on the Diffusion-LM family of implementations. We thank
the authors of the upstream repositories.
