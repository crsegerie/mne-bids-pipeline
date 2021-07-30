"""In this file we analyse the contrast insource space.

- Apply inverse operator to covariances of different conditions.
- We substract the source space between the two conditions.


Inspired from: 
https://mne.tools/stable/auto_examples/inverse/mne_cov_power.html

(mne_dev) csegerie@drago2:~/Desktop/mne-bids-pipeline$ nice -n 5 xvfb-run  python run.py --config=/storage/store2/data/time_in_wm_new/derivatives/decoding/cfg.py --steps=source/06-source_contrast
"""

import itertools
import logging
from typing import Optional

import mne
from mne.epochs import read_epochs
from mne.minimum_norm.inverse import apply_inverse_cov
from mne.utils import BunchConst
from mne.parallel import parallel_func
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              read_inverse_operator)
from mne_bids import BIDSPath

import config
from config import gen_log_kwargs, on_error, failsafe_run, sanitize_cond_name


logger = logging.getLogger('mne-bids-pipeline')


def one_subject(subject, session, cfg):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         recording=cfg.rec,
                         space=cfg.space,
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root,
                         check=False)

    fname_inv = bids_path.copy().update(suffix='inv')
    inverse_operator = read_inverse_operator(fname_inv)

    fname_epoch = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        suffix='epo',
        extension='.fif',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        processing='clean',
        check=False)

    epochs = read_epochs(fname_epoch)
    print("epochs loaded")
    epochs.decimate(5)

    # apres le faire pour les deux contrastes
    base_cov = mne.compute_covariance(
        epochs, tmin=-0.2, tmax=0, method=['shrunk', 'empirical'])
    data_cov = mne.compute_covariance(
        epochs, tmin=0., tmax=0.2, method=['shrunk', 'empirical'])

    stc_data = apply_inverse_cov(data_cov, epochs.info, inverse_operator,
                                 nave=len(epochs), method='dSPM', verbose=True)
    stc_base = apply_inverse_cov(base_cov, epochs.info, inverse_operator,
                                 nave=len(epochs), method='dSPM', verbose=True)

    stc_data /= stc_base  # type: ignore
    brain = stc_data.plot(
        subjects_dir=config.get_fs_subjects_dir(),
        clim=dict(kind='percent', lims=(50, 90, 98)))
    brain.save_image(filename="test.png", mode='rgb')


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> BunchConst:
    cfg = BunchConst(
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        ch_types=config.ch_types,
        conditions=config.conditions,
        inverse_method=config.inverse_method,
        deriv_root=config.get_deriv_root(),
        subjects_dir=config.get_fs_subjects_dir()
    )
    return cfg


def main():
    subject = config.get_subjects()[0]
    session = config.get_sessions()[0]
    one_subject(subject=subject, session=session, cfg=get_config())


if __name__ == '__main__':
    main()
