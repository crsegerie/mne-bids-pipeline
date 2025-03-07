"""
===========
04. Run SSP
===========

Compute Signal Subspace Projections (SSP).
"""

import itertools
import logging
from typing import Optional

import mne
from mne.utils import BunchConst
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.parallel import parallel_func
from mne_bids import BIDSPath

import config
from config import gen_log_kwargs, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error, script_path=__file__)
def run_ssp(*, cfg, subject, session=None):
    # compute SSP on first run of raw
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=cfg.runs[0],
                         recording=cfg.rec,
                         space=cfg.space,
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root)

    # Prepare a name to save the data
    raw_fname_in = bids_path.copy().update(processing='filt', suffix='raw',
                                           check=False)

    # when saving proj, use run=None
    proj_fname_out = bids_path.copy().update(run=None, suffix='proj',
                                             check=False)

    msg = f'Input: {raw_fname_in}, Output: {proj_fname_out}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    if raw_fname_in.copy().update(split='01').fpath.exists():
        raw_fname_in.update(split='01')

    raw = mne.io.read_raw_fif(raw_fname_in)
    msg = 'Computing SSPs for ECG'
    logger.debug(**gen_log_kwargs(message=msg, subject=subject,
                                  session=session))

    reject_ecg_ = config.get_ssp_reject(
        ssp_type='ecg',
        epochs=(create_ecg_epochs(raw)
                if cfg.ssp_reject_ecg == 'autoreject_global' else None))
    ecg_projs, _ = compute_proj_ecg(raw,
                                    average=cfg.ecg_proj_from_average,
                                    reject=reject_ecg_,
                                    **cfg.n_proj_ecg)
    if not ecg_projs:
        msg = 'No ECG events could be found. No ECG projectors computed.'
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session))

    msg = 'Computing SSPs for EOG'
    logger.debug(**gen_log_kwargs(message=msg, subject=subject,
                                  session=session))
    if cfg.eog_channels:
        ch_names = cfg.eog_channels
        assert all([ch_name in raw.ch_names for ch_name in ch_names])
    else:
        ch_names = None

    reject_eog_ = config.get_ssp_reject(
        ssp_type='eog',
        epochs=(create_eog_epochs(raw)
                if cfg.ssp_reject_eog == 'autoreject_global' else None))
    eog_projs, _ = compute_proj_eog(raw, ch_name=ch_names,
                                    average=cfg.eog_proj_from_average,
                                    reject=reject_eog_,
                                    **cfg.n_proj_eog)

    if not eog_projs:
        msg = 'No EOG events could be found. No EOG projectors computed.'
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session))

    mne.write_proj(proj_fname_out, eog_projs + ecg_projs)


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> BunchConst:
    cfg = BunchConst(
        runs=config.get_runs(subject=subject),
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        eog_channels=config.eog_channels,
        deriv_root=config.get_deriv_root(),
        ssp_reject_ecg=config.ssp_reject_ecg,
        ecg_proj_from_average=config.ecg_proj_from_average,
        ssp_reject_eog=config.ssp_reject_eog,
        eog_proj_from_average=config.eog_proj_from_average,
        n_proj_eog=config.n_proj_eog,
        n_proj_ecg=config.n_proj_ecg,
    )
    return cfg


def main():
    """Run SSP."""
    if not config.spatial_filter == 'ssp':
        msg = 'Skipping …'
        logger.info(**gen_log_kwargs(message=msg))
        return

    parallel, run_func, _ = parallel_func(run_ssp,
                                          n_jobs=config.get_n_jobs())
    logs = parallel(
        run_func(cfg=get_config(subject=subject), subject=subject,
                 session=session)
        for subject, session in
        itertools.product(config.get_subjects(),
                          config.get_sessions())
    )

    config.save_logs(logs)


if __name__ == '__main__':
    main()
