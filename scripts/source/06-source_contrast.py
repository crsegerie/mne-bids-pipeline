"""In this file we analyse the contrast insource space.

- Apply inverse operator to covariances of different conditions.
- We substract the source space between the two conditions.


Inspired from: 
https://mne.tools/stable/auto_examples/inverse/mne_cov_power.html





    stc_data = apply_inverse_cov(data_cov, evoked.info, inverse_operator,
                             nave=len(epochs), method='dSPM', verbose=True)
    stc_base = apply_inverse_cov(base_cov, evoked.info, inverse_operator,
                                nave=len(epochs), method='dSPM', verbose=True)



"""

import itertools
import logging
from typing import Optional

import mne
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
    )
    return cfg


def main():
    subject = config.get_subjects()[0]
    session = config.get_sessions()[0]
    one_subject(subject=subject, session=session, cfg=get_config())


if __name__ == '__main__':
    main()
