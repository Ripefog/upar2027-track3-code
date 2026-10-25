#!/usr/bin/env python3
"""Convert PoseTrack18 per-video annotations (train+val) to one COCO-style train_posetrack.json.

PoseTrack18 already uses the challenge's 17 slots (nose, head_bottom, head_top, ears, ...), so no
re-mapping is needed. Ears (slots 3-4) are zeroed because the challenge never scores them. Only frames
with is_labeled=True are kept. A person needs >= --min-joints labelled joints (v>0 and not at (0,0)).
Boxes come from the keypoint extent (PoseTrack 'bbox' is used when present and sane).
"""
import argparse, glob, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels-root', required=True, help='dir containing annotations/{train,val}')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--image-prefix', default='posetrack')
    ap.add_argument('--min-joints', type=int, default=5)
    ap.add_argument('--min-box-height', type=float, default=40.)
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    args = ap.parse_args()
    images, anns, kept, dropped, uid = [], [], 0, 0, 0
    for split in args.splits:
        for f in sorted(glob.glob(f'{args.labels_root}/annotations/{split}/*.json')):
            d = json.load(open(f))
            labeled = {im['id'] for im in d['images'] if im.get('is_labeled')}
            for im in d['images']:
                if im['id'] in labeled:
                    images.append({'id': int(im['id']), 'file_name': f"{args.image_prefix}/{im['file_name']}",
                                   'width': im.get('width', 0), 'height': im.get('height', 0)})
            for a in d['annotations']:
                if a['image_id'] not in labeled:
                    continue
                kp = [float(v) for v in a['keypoints']]
                if len(kp) != 51:
                    dropped += 1; continue
                for j in (3, 4):
                    kp[j*3:j*3+3] = [0., 0., 0.]
                for j in range(17):
                    if kp[j*3+2] > 0 and kp[j*3] == 0 and kp[j*3+1] == 0:
                        kp[j*3+2] = 0.
                    if kp[j*3+2] <= 0:
                        kp[j*3:j*3+3] = [0., 0., 0.]
                vis = [(kp[j*3], kp[j*3+1]) for j in range(17) if kp[j*3+2] > 0]
                box = a.get('bbox') or []
                if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
                    if not vis:
                        dropped += 1; continue
                    xs, ys = zip(*vis)
                    box = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                ok = len(vis) >= args.min_joints and box[3] >= args.min_box_height
                if not ok:
                    kp = [0.] * 51; dropped += 1
                else:
                    kept += 1
                uid += 1
                anns.append({'id': uid, 'image_id': int(a['image_id']), 'category_id': 1, 'bbox': [float(v) for v in box],
                             'area': float(box[2] * box[3]), 'iscrowd': 0, 'keypoints': kp,
                             'num_keypoints': sum(kp[j*3+2] > 0 for j in range(17))})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / 'train_posetrack.json').write_text(json.dumps(
        {'images': images, 'annotations': anns, 'categories': [{'id': 1, 'name': 'person'}]}))
    print(f'labelled_images={len(images)} usable_persons={kept} ignored={dropped} -> {out/"train_posetrack.json"}')

if __name__ == '__main__':
    main()
