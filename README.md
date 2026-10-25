# UPAR Challenge 2027 — Track 3: Top-down 2D Human Pose Estimation

**Code submission for verification by the organisers.**
The final submission reached **hidden AP 0.9065** (mean OKS AP 0.50:0.95, four hidden subsets, maxDets = 20) on Codabench, using a **pure ViTPose-H fine-tuned from the official single-dataset COCO checkpoint**.

## Final leaderboard result

| Final submission ZIP | Codabench hidden AP |
|---|---:|
| `vith_cocoonly_soup_llrd_mh_meanpeak_flipshift_dark.zip` | **0.9065** |

Mean OKS AP 0.50:0.95 over the four hidden subsets, maxDets = 20; scoring time 2,385 s for 30,245 boxes. The submitted weights use `score_mode: mean_peak` (the mean raw heatmap peak over the 15 scored joints) as the instance score.

sha256 of the submitted ZIP: `5b17095abac9d7caed3cca9a333182309769be81893937f488218c681891fd33`

## Trained weights

The trained weights of the final submission — the uniform soup `best.pth` that was packed into the ZIP above — are attached to the GitHub release:

**https://github.com/Ripefog/upar2027-track3-code/releases/tag/final-weights**

GitHub caps release assets at 2 GiB, so the file is shipped in two parts. Rejoin and verify:

```bash
cat soup_best.pth.part-00 soup_best.pth.part-01 > best.pth
sha256sum best.pth   # expected: 6757793f8eedc89855a87abc47a0f7b5534f8ee2fa7d3de26228796f642fd4d3
```

Place the rejoint file at `outputs/vith_cocoonly_soup_llrd_mh/best.pth` and run step 4 of `scripts/reproduce_submission.sh` to pack a submission ZIP with identical weights and predictions — no retraining needed.

## Method summary

- **Model:** ViTPose-H (~637 M params), 256×192 input, 17 PoseTrack18 slots, 15 scored joints (ears unscored), initialised from the official single-dataset COCO ViTPose-H checkpoint (converted with `scripts/convert_vitpose_official.py`).
- **Fine-tuning (2 epochs, AdamW cosine, EMA 0.999, domain-balanced sampling, seed 2027):**
  - *Variant A:* head-loss masking on PoseTrack (`--mask-head-domains posetrack`) — PoseTrack head labels follow a different annotation convention, so the head rows of the loss are masked there (auto-generated head rows are kept for the other domains).
  - *Variant B:* A + layer-wise LR decay 0.9 and stochastic depth (drop-path 0.2).
- **Model soup:** uniform weight average of A and B (valid because both share seed 2027 and the same head initialisation).
- **Inference:** flip TTA with a 1-cell heatmap shift (`--flip-tta`, shift 1), DARK heatmap decoding (kernel 7), and mean-peak instance scoring. Inference time on the hidden server: 2,385 s for 30,245 boxes.
- **Selection protocol:** all model/config selection used the source validation split (image-hash 10 % holdout, seed 2027) and leave-one-domain-out experiments. The hidden test was never used for training, pseudo-labelling, tuning, or selection.

## Repository layout

```
Dockerfile                   # pinned environment (pytorch 2.13.0 / CUDA 13.0), self-checks in build
train.py                     # training entry point (fine-tune, EMA, soup ingredients, provenance)
pose_src/                    # models, data pipeline, geometry (DARK, flip+shift), metrics
configs/                     # all experiment configs (the two soup ingredients:
                             #   vitpose_h_mix_ft.yaml = variant A, vitpose_h_llrd_dp.yaml = variant B)
scripts/
  reproduce_submission.sh    # exact end-to-end chain for the final ZIP
  convert_vitpose_official.py  make_soup.py  pack_submission.py  eval_submission.py
  prepare_*.py               # dataset preparation helpers
  score_study.py tta_study.py bias_study.py analyze_*.py compare_*.py paper_*.py
                             # validation-time studies backing the design choices (source data only)
submission/                  # the code that ships inside the ZIP (run.py, metadata.yaml)
provenance/                  # configs, training histories, packing/eval logs and models.json
                             # of the final submission (evidence for verification)
local/provenance_vitpose_h_cocoonly.txt   # pretraining provenance statement (required by train.py)
tests/                       # unit tests (mock data) + rules self-check
requirements_train.txt       # minimal training dependencies (install torch separately, see below)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
# PyTorch for the GPU used (RTX PRO 6000 Blackwell):
pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements_train.txt
pytest tests/          # mock-data unit tests, incl. submission smoke test
```

Training data: UPAR source training images + COCO-format annotations (Market-1501, PA-100K, PETA, PoseTrack18, COCO keypoints), obtained through the official Track 3 starter kit (`download_datasets.py`); PoseTrack18 images referenced by the annotations are extracted with `scripts/prepare_posetrack18.py`. The official ViTPose-H COCO checkpoint (`SingleCOCO/vitpose-h.pth`) is downloaded manually from the ViTPose author release and converted once:

```bash
python scripts/convert_vitpose_official.py <SingleCOCO/vitpose-h.pth> weights/vitpose_h_cocoonly_native.pth
```

## Docker

```bash
docker build -t upar-track3 .
docker run --gpus all -it --rm \
  -v /path/to/annotations:/ann -v /path/to/images:/images \
  upar-track3 bash
# inside the container:
ANN=/ann IMG=/images bash scripts/reproduce_submission.sh
```

The image is based on `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime` — the same torch/CUDA combination as the training environment of the final submission — and runs the mock-data unit tests during the build.

## Reproducing the final submission

Full pipeline (trains variant A and B with the exact command lines of the submitted run, averages them into a uniform soup, packs and evaluates the ZIP):

```bash
ANN=/path/to/upar/annotations IMG=/path/to/images PY=python bash scripts/reproduce_submission.sh
```

The script patches the annotation dir / image root into the configs automatically and trains on the exact domain mix used for the submission (`market pa100k peta posetrack coco`, with PoseTrack head-loss masking). To skip training, download the released weights (see "Trained weights") and run only step 4. The configs, training histories and logs of the submitted run are in `provenance/`.

## Compliance notes

- Initialisation is the official COCO-only ViTPose-H release; no additional pretraining was performed by us. sha256 of the downloaded official checkpoint: `092b137f18fb92ec0c569dd9d003468e4a31fbd9248fe4523cd9f10a15ad17a6`; of the converted native checkpoint used by `train.py`: `189c1d49aa21ec6e99f455d4379e89ee8018de1f0b39912e492be93fb64906d9`.
- The hidden test data was never used for training, pseudo-labelling, tuning or model selection.
- `tests/test_rules_2027.py` re-checks the repo's own compliance assumptions.

## License

MIT (see `LICENSE`), following the Track 3 benchmark template.
