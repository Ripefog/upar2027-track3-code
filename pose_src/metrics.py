"""Local source-domain AP using pycocotools keypoint evaluation.

NOT a copy of the challenge's private scorer. It implements public COCOeval
with 17 slots, ear ground-truth masks, maxDets=20, and source-domain averaging.
Do not interpret a local source-domain AP as hidden-test AP.
"""
from __future__ import annotations
import contextlib
import io
import json
import math
from pathlib import Path

from .geometry import JOINTS


def coco_ap_for_domain(annotation_path, predictions, validation_image_ids, verbose=False):
    """Evaluate source-domain predictions against *allowed training annotations*.

    predictions: list of COCO person predictions with image_id, keypoints (51),
    category_id=1 and score. Ignore/crowd GT remain for COCO bookkeeping.
    Returns AP, AP50, AP_medium, AP_large; invalid -1 metrics become None.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as ex:
        raise RuntimeError('Local OKS AP requires pycocotools. Install: pip install pycocotools') from ex
    allowed_ids = set(int(n) for n in validation_image_ids)
    if not allowed_ids:
        raise ValueError('Cannot compute AP with no validation image IDs')
    if not predictions:
        raise ValueError('Cannot compute AP with no predictions')
    raw = json.loads(Path(annotation_path).read_text(encoding='utf-8'))
    images = [im for im in raw['images'] if int(im['id']) in allowed_ids]
    anns = []
    for original in raw['annotations']:
        if int(original['image_id']) not in allowed_ids:
            continue
        ann = dict(original)
        k = ann.get('keypoints', [0.] * 51)
        if len(k) != 51:
            k = [0.] * 51
        k = list(k)
        for j in (3, 4):
            k[j * 3 + 2] = 0.
        ann['keypoints'] = k
        ann['num_keypoints'] = sum(k[j * 3 + 2] > 0 for j in range(17))
        box = ann.get('bbox', [0., 0., 0., 0.])
        ann['area'] = float(ann.get('area', max(0., box[2]) * max(0., box[3])))
        ann['iscrowd'] = int(ann.get('iscrowd', 0))
        ann['category_id'] = 1
        anns.append(ann)
    ground_truth = {'info': raw.get('info', {}), 'licenses': raw.get('licenses', []),
                    'images': images, 'annotations': anns,
                    'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person',
                                    'keypoints': list(JOINTS), 'skeleton': []}]}
    if not images or not anns:
        raise ValueError('No GT annotations in selected domain/validation fold')
    # COCO.loadRes indexes result annotations by list position; filter invalid
    # image IDs and retain one scored person result per input box.
    detections = [p for p in predictions if int(p['image_id']) in allowed_ids]
    if not detections:
        raise ValueError('Validation predictions list is empty for the selected domain')
    with contextlib.redirect_stdout(io.StringIO() if not verbose else __import__('sys').stdout):
        gt = COCO()
        gt.dataset = ground_truth
        gt.createIndex()
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, iouType='keypoints')
        ev.params.imgIds = sorted(allowed_ids)
        ev.params.catIds = [1]
        ev.params.maxDets = [20]
        # COCOeval defaults to COCO keypoint sigmas. These may differ from
        # the private challenge's exact evaluator settings; verify if released.
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    keys = ('AP', 'AP50', 'AP75', 'AP_medium', 'AP_large')
    vals = (ev.stats[0], ev.stats[1], ev.stats[2], ev.stats[3], ev.stats[4])
    result = {k: (float(v) if math.isfinite(float(v)) and v >= 0 else None)
              for k, v in zip(keys, vals)}
    result['images'] = len(images)
    result['ground_truth_instances'] = len(anns)
    result['predictions'] = len(detections)
    return result
