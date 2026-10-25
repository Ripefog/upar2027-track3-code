"""Track 3 format/safety regressions, no private images/weights needed."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pose_src.geometry import SCORED_INDEX, calibrated_joint_confidence, person_confidence
from pose_src.data import PoseDataset
from pose_src.models import build_model


ROOT=Path(__file__).resolve().parents[1]


def test_scored_joint_slots_and_confidence():
    assert list(SCORED_INDEX)==[0,1,2,*list(range(5,17))]
    raw=np.zeros(17,np.float32)
    raw[[3,4]]=1e6
    a=float(person_confidence(raw))
    raw[[3,4]]=-1e6
    assert float(person_confidence(raw))==pytest.approx(a)
    assert float(person_confidence(np.ones(17))) > a
    assert np.isfinite(calibrated_joint_confidence(np.array([-1e6,1e6]))).all()


def test_submission_metadata_exact_expected_lines():
    meta=(ROOT/'submission'/'metadata.yaml').read_text()
    assert meta.startswith('# Code submissions are executed by the challenges ingestion program')
    assert 'command: python3 -c' in meta


def test_pack_and_serve_tiny_mock(tmp_path):
    # Random weights are for testing wiring only; NEVER upload this to Codabench.
    cfg={'model':{'name':'vitpose','variant':'tiny'}, 'data':{'input_size':[64,48]}}
    state=build_model(cfg).state_dict()
    ckpt=tmp_path/'synthetic_mock_NOT_FOR_SUBMISSION.pth'
    torch.save({'state_dict':state,'config':cfg},ckpt)
    target=tmp_path/'mock_DO_NOT_SUBMIT.zip'
    subprocess.run([sys.executable,str(ROOT/'scripts'/'pack_submission.py'),
                    '--checkpoints',str(ckpt),'--output',str(target)],check=True,capture_output=True)
    import zipfile
    with zipfile.ZipFile(target) as z:
        assert 'metadata.yaml' in z.namelist()
        assert 'run.py' in z.namelist()
        assert 'models.json' in z.namelist()
    subprocess.run([sys.executable,str(ROOT/'scripts'/'smoke_submission.py'),
                    '--submission',str(target)],check=True,capture_output=True)
