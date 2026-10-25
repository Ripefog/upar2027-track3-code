#!/usr/bin/env python3
"""Convert COCO person_keypoints_train2017.json to the 17-slot PoseTrack18 schema.

COCO has no head_bottom/head_top, so slots 1-2 get v=0 (never supervised during
COCO pretraining). COCO eyes/ears are DROPPED: eyes are not head_bottom/head_top
and ears are unscored. Slots 0 and 5-16 keep the same index in both layouts.
Persons with < --min-joints labelled joints or a tiny box have keypoints zeroed,
so COCOeval ignores them and PoseDataset skips them.
"""
import argparse
import json
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotations-zip', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--image-prefix', default='coco/train2017')
    ap.add_argument('--min-joints', type=int, default=5)
    ap.add_argument('--min-box-height', type=float, default=40.)
    args = ap.parse_args()
    with zipfile.ZipFile(args.annotations_zip) as z:
        raw = json.loads(z.read('annotations/person_keypoints_train2017.json'))
    images = []
    for im in raw['images']:
        im = dict(im)
        im['file_name'] = f"{args.image_prefix}/{im['file_name']}"
        images.append(im)
    kept = dropped = 0
    anns = []
    for a in raw['annotations']:
        src = a['keypoints']
        kp = [0.] * 51
        for j in [0, *range(5, 17)]:
            kp[j*3:j*3+3] = src[j*3:j*3+3]
        n = sum(kp[j*3+2] > 0 for j in range(17))
        ok = (not a['iscrowd']) and n >= args.min_joints and a['bbox'][3] >= args.min_box_height
        if not ok:
            kp = [0.] * 51; n = 0; dropped += 1
        else:
            kept += 1
        anns.append({'id': a['id'], 'image_id': a['image_id'], 'category_id': 1,
                     'bbox': a['bbox'], 'area': a['area'], 'iscrowd': a['iscrowd'],
                     'keypoints': kp, 'num_keypoints': n})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / 'train_coco.json').write_text(json.dumps(
        {'images': images, 'annotations': anns,
         'categories': [{'id': 1, 'name': 'person'}]}))
    print(f'images={len(images)} usable_persons={kept} ignored={dropped} -> {out/"train_coco.json"}')


if __name__ == '__main__':
    main()
