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
from mne.source_estimate import SourceEstimate
import numpy as np

import mne
from mne.epochs import BaseEpochs, read_epochs
from mne.minimum_norm.inverse import apply_inverse_cov
from mne.utils import BunchConst
from mne.parallel import parallel_func
from mne.minimum_norm import read_inverse_operator
from mne_bids import BIDSPath

import config


logger = logging.getLogger('mne-bids-pipeline')


def fname(subject, session):
    """Get name of source file."""
    fname = f"res/brain_contrast_morphed_sub-{subject}-ses-{session}.stc"
    return fname


def one_subject(subject, session, cfg):
    """Compute the contrast and morph it to the fsavg."""
    bids_path = BIDSPath(
        subject=subject,
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

    stc_cond = []
    for cond in config.contrasts[0]:  # type: ignore
        print(cond)
        l_freq, h_freq = 8, 12

        epochs_filter: BaseEpochs = epochs[cond]  # type: ignore
        base_epochs = epochs_filter.copy().crop(tmin=-0.2, tmax=0)
        base_epochs.filter(l_freq, h_freq)
        data_epochs = epochs_filter.copy().crop(tmin=0, tmax=0.5)
        data_epochs.filter(l_freq, h_freq)

        base_cov = mne.compute_covariance(base_epochs)
        data_cov = mne.compute_covariance(data_epochs)

        stc_data = apply_inverse_cov(
            data_cov, epochs.info, inverse_operator,
            nave=len(epochs), method='dSPM', verbose=True)
        stc_base = apply_inverse_cov(
            base_cov, epochs.info, inverse_operator,
            nave=len(epochs), method='dSPM', verbose=True)

        stc_data /= stc_base  # type: ignore
        stc_cond.append(stc_data)
        brain = stc_data.plot(subjects_dir=config.get_fs_subjects_dir())
        brain_img = f"res/brain_{cond}-sub-{subject}-ses-{session}.png"
        brain.save_image(filename=brain_img, mode='rgb')

    stc_contrast = stc_cond[1] - stc_cond[0]

    morph = mne.compute_source_morph(
        stc_contrast,
        subject_from=config.get_fs_subject(subject), subject_to='fsaverage',
        subjects_dir=cfg.fs_subjects_dir)
    stc_fsaverage: SourceEstimate = morph.apply(stc_contrast)  # type: ignore

    brain = stc_fsaverage.plot(
        subjects_dir=config.get_fs_subjects_dir(), hemi="split")
    brain.save_image(
        filename=f"res/brain_contrast_morphed_sub-{subject}-ses-{session}.png",
        mode='rgb')

    stc_fsaverage.save(fname=fname(subject, session))

    return stc_fsaverage


def group_analysis(subjects, sessions, cfg):
    """Complete group analysis."""
    tab_stc_fsaverage = [[None]*len(sessions)] * len(subjects)
    for sub, subject in enumerate(subjects):
        for ses, session in enumerate(sessions):
            tab_stc_fsaverage[sub][ses] = mne.read_source_estimate(
                fname=fname(subject, session), subject=subject)
    stc_avg = np.array(tab_stc_fsaverage).mean()

    subject = "fsaverage"
    stc_avg.subject = subject
    brain = stc_avg.plot(
        subjects_dir="/storage/store2/data/time_in_wm_new/derivatives/freesurfer/subjects",
        hemi="split")
    brain.save_image(
        filename=f"res/brain_contrast_morphed_sub-{subject}.png",
        mode='rgb')


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
        fs_subjects_dir=config.get_fs_subjects_dir()
    )
    return cfg


def main():
    """Source space contrast."""
    subjects = config.get_subjects()
    print(subjects)
    sessions = config.get_sessions()
    cfg = get_config()

    # one_subject(subject=subject, session=session, cfg=cfg)

    parallel, run_func, _ = parallel_func(one_subject,
                                          n_jobs=config.get_n_jobs())
    parallel(
        run_func(cfg=cfg, subject=subject, session=session)
        for subject, session in
        itertools.product(subjects, sessions)
    )
    group_analysis(subjects, sessions, cfg)


if __name__ == '__main__':
    main()
