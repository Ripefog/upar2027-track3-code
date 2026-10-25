#!/usr/bin/env bash
# Reproduce the final Codabench submission (hidden AP 0.9065):
#   vith_cocoonly_soup_llrd_mh_meanpeak_flipshift_dark.zip
# Pure ViTPose-H initialised from the official single-dataset COCO checkpoint.
# Hardware used: 1x NVIDIA RTX PRO 6000 Blackwell (96 GB); both trainings together ~6 h.
set -euo pipefail

PY=${PY:-python}        # training env: torch 2.13.0 + cu130, see requirements_train.txt / Dockerfile
ANN=${ANN:?Set ANN to the UPAR source annotation dir containing train_<domain>.json for market/pa100k/peta/posetrack/coco}
IMG=${IMG:-data}        # image root of the source datasets
CKPT=${CKPT:-weights/vitpose_h_cocoonly_native.pth}

# Patch the annotation dir / image root into both configs (no manual yaml editing needed)
WORK=work; mkdir -p $WORK
for c in configs/vitpose_h_mix_ft.yaml configs/vitpose_h_llrd_dp.yaml; do
  sed -e "s#annotation_dir: .*#annotation_dir: $ANN#" -e "s#image_root: .*#image_root: $IMG#" \
      "$c" > "$WORK/$(basename "$c")"
done
DOMAINS="market pa100k peta posetrack coco"

# 0) one-time: convert the official ViTPose-H COCO checkpoint (origin + sha256 in README.md)
#    $PY scripts/convert_vitpose_official.py <official SingleCOCO/vitpose-h.pth> $CKPT
#    No-retraining alternative: download the released final soup weights, rejoin them
#    (see README "Trained weights") and place at
#    outputs/vith_cocoonly_soup_llrd_mh/best.pth, then jump to step 4.

# 1) Variant A: fine-tune + PoseTrack head-loss masking (mask_head_domains: posetrack)
$PY train.py --config $WORK/vitpose_h_mix_ft.yaml \
    --pretrained $CKPT --pretrained-topology coco \
    --provenance local/provenance_vitpose_h_cocoonly.txt \
    --domains $DOMAINS --mask-head-domains posetrack \
    --output outputs/vith_cocoonly_maskhead_mix_ft

# 2) Variant B: same recipe + layer-wise LR decay 0.9 + drop-path 0.2
#    (same seed 2027 and same head initialisation -> soup-compatible with A)
$PY train.py --config $WORK/vitpose_h_llrd_dp.yaml \
    --pretrained $CKPT --pretrained-topology coco \
    --provenance local/provenance_vitpose_h_cocoonly.txt \
    --domains $DOMAINS --mask-head-domains posetrack \
    --output outputs/vith_cocoonly_llrd_dp_mix_ft

# 3) Uniform weight soup of A and B
$PY scripts/make_soup.py outputs/vith_cocoonly_soup_llrd_mh/best.pth \
    outputs/vith_cocoonly_maskhead_mix_ft/best.pth \
    outputs/vith_cocoonly_llrd_dp_mix_ft/best.pth

# 4) Pack the submission: flip TTA with 1-cell heatmap shift, DARK decoding,
#    mean-peak instance score (score_mode=mean_peak)
$PY scripts/pack_submission.py \
    --checkpoints outputs/vith_cocoonly_soup_llrd_mh/best.pth \
    --output dist/vith_cocoonly_soup_llrd_mh_meanpeak_flipshift_dark.zip \
    --flip-tta --batch-size 32 --decoder dark --score-mode mean_peak

# 5) Optional: local source-domain evaluation of the packed ZIP
$PY scripts/eval_submission.py \
    --submission dist/vith_cocoonly_soup_llrd_mh_meanpeak_flipshift_dark.zip \
    --annotation-dir $ANN --image-root $IMG \
    --output outputs/eval_soup_llrd_mh_meanpeak.eval.json
