"""In this file, we analyze the contrast in the source space.

- We apply the inverse operator to the covariances for different conditions.
- We subtract the source estimates between the two conditions.
- We average across subjects this contrast.

For now, the inverse operator uses the empty room covariance.

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


def plot_source(stc, filename):
    """Plot and save the source estimate."""
    brain = stc.plot(
        subjects_dir=config.get_fs_subjects_dir(),
        hemi="split", size=(1600, 800))
    brain.save_image(filename=filename, mode='rgb')


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
        l_freq, h_freq = 8, 14

        epochs_filter: BaseEpochs = epochs[cond]  # type: ignore
        data_epochs = epochs_filter.copy().crop(tmin=0, tmax=3)
        data_epochs.filter(l_freq, h_freq)

        data_cov = mne.compute_covariance(data_epochs)

        # Here the inverse uses the empty room recording.
        stc_data = apply_inverse_cov(
            data_cov, epochs.info, inverse_operator,
            nave=len(epochs), method='dSPM', verbose=False)
        print("subject", subject, np.max(stc_data.data), np.min(stc_data.data))
        stc_data.data = np.log(stc_data.data)
        stc_cond.append(stc_data)
        filename = f"res/brain_{cond}-sub-{subject}-ses-{session}.png"
        plot_source(stc_data, filename)

    # Taking the difference of the log
    stc_contrast = stc_cond[1] - stc_cond[0]

    filename = f"res/brain_contrast_sub-{subject}-ses-{session}.png"
    plot_source(stc_contrast, filename)

    morph = mne.compute_source_morph(
        stc_contrast,
        subject_from=config.get_fs_subject(subject), subject_to='fsaverage',
        subjects_dir=cfg.fs_subjects_dir)
    stc_fsaverage: SourceEstimate = morph.apply(stc_contrast)  # type: ignore

    filename = f"res/brain_contrast_morphed_sub-{subject}-ses-{session}.png"
    plot_source(stc_fsaverage, filename)

    stc_fsaverage.save(fname=fname(subject, session))

    return stc_fsaverage


def group_analysis(subjects, sessions, cfg):
    """Take the average of the source estimates."""
    tab_stc_fsaverage = [[None]*len(sessions)] * len(subjects)
    for sub, subject in enumerate(subjects):
        for ses, session in enumerate(sessions):
            tab_stc_fsaverage[sub][ses] = mne.read_source_estimate(
                fname=fname(subject, session), subject=subject)
    stc_avg = np.array(tab_stc_fsaverage).mean()

    # TODO: Not elegant
    subject = "fsaverage"
    stc_avg.subject = subject
    brain = stc_avg.plot(
        subjects_dir="/storage/store2/data/time_in_wm_new/derivatives/freesurfer/subjects",
        hemi="split", size=(1600, 800))
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

    # Usefull for debugging
    # for sub, ses in itertools.product(subjects, sessions):
    #     one_subject(sub, ses, cfg)

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
