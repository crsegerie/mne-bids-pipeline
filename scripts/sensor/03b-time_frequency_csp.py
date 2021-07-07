"""
====================================================================
Decoding in time-frequency space using Common Spatial Patterns (CSP)
====================================================================

This file contains two main steps:
- 1. Decoding
The time-frequency decomposition is estimated by iterating over raw data that
has been band-passed at different frequencies. This is used to compute a
covariance matrix over each epoch or a rolling time-window and extract the CSP
filtered signals. A linear discriminant classifier is then applied to these
signals. More detail are available here:
https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html#common-spatial-pattern
Warning: This step, especially the double loop on the time-frequency bins
is very computationnaly expensive.

- 2. Permutation statistics
We try to answer the following question: is the difference between
the two conditions statistically significant? We use the classic permutations
cluster tests on the time-frequency roc-auc map.
More détails are available here:
https://mne.tools/stable/auto_tutorials/stats-sensor-space/10_background_stats.html#sphx-glr-auto-tutorials-stats-sensor-space-10-background-stats-py

The user has only to specify the [f_min, f_max] band
with the n_freq wich is the number of frequency windows + 1.
# TODO : bad api.
And we calculate automatically the time-windows based on the Nyquist criterion.
"""
# License: BSD (3-clause)

from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, Union
import logging
from matplotlib.figure import Figure

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm  # TODO does not work smoothly

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from mne.stats.cluster_level import permutation_cluster_1samp_test
from mne.epochs import BaseEpochs
from mne import create_info, read_epochs, set_log_level
from mne.decoding import CSP
from mne.time_frequency import AverageTFR
from mne.parallel import parallel_func
from mne.utils import BunchConst
from mne.report import Report

from mne_bids import BIDSPath

# mne-bids-pipeline config
from config import N_JOBS, gen_log_message, on_error, failsafe_run
import config

logger = logging.getLogger('mne-bids-pipeline')
set_log_level(verbose="WARNING")  # mne logger


# TODO : Should we generalize this with the parameter decoding_metric?
# ROC-AUC chance score level
chance = 0.5


class Pth:
    """Util class containing usefull Paths infos."""

    def __init__(self, cfg) -> None:
        """Initialize directory. Initialize the base path."""
        # We initialise to the average subject and session ?
        self.bids_basename = BIDSPath(
            subject="average",
            # session=config.get_sessions()[0], # TODO
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

        # TODO remove from here
        # create directory if it does not exist
        Path("results").mkdir(parents=True, exist_ok=True)
        Path("results/npy").mkdir(parents=True, exist_ok=True)

        self.report_fname = self.bids_basename.copy().update(
            processing='csp+permutation+test',
            suffix='report',
            extension='.html')

    def file(
        self,
        *,
        subject: str,
        # TODO: Not DRY and Any != Unkwnown
        # TODO: Does not work if multiple session
        session: Union[list[None],  str, List[Any]],
        cfg
    ) -> BIDSPath:
        """Return the path of the file."""
        return self.bids_basename.copy().update(subject=subject)

    def freq_scores(self, subject):
        return f"results/npy/freq_scores-{subject}.npy"

    def freq_scores_std(self, subject):
        return f"results/npy/freq_scores_std-{subject}.npy"

    def tf_scores(self, subject):
        return f"results/npy/tf_scores-{subject}.npy"


def compute_n_time_windows(cfg) -> int:  # This could go in the tf class
    """Compute the number of time windows.

    We begin by calculating the minimum Nyquist windows size according
    to the hiher frequence. Then we take n_cycle times this mininum duration.
    """
    # Infer window spacing from the min freq and number of cycles to avoid gaps
    # For band passed periodic signal, according to the Nyquist theorem,
    # we can reconstruct the signal if f_s > 2 * band_freq
    band_freq = (cfg.time_frequency_freq_max -
                 cfg.time_frequency_freq_min)/(cfg.n_freqs - 1)
    min_nyquist_period = 1 / (2 * band_freq)

    # But the signal should be periodic.
    # One period of the signal is not suffiscient.
    # So we recommend to take at least n_cycles > 10
    min_window_size = min_nyquist_period * cfg.n_cycles
    msg = (f"The minimum Nyquist windows period is "
           f"{round(min_nyquist_period, 3)}s, "
           f"n_cycles={cfg.n_cycles} which puts your time windows "
           f"size to a minimum of {round(min_window_size, 3)}s.")
    logger.info(gen_log_message(message=msg, step=3))

    # TODO: no more n_cycle
    # In oder to count how many min_asked_spacing windows we can fit in the
    # epochs range, we use the np.arange method.
    centered_w_times_not_adjusted = \
        np.arange(cfg.epochs_tmin, cfg.epochs_tmax, min_window_size)[1:]

    n_time_windows = len(centered_w_times_not_adjusted)
    if n_time_windows == 0:
        msg = ("Minimum windows size do not fit in the epoch interval. "
               "Try to increase your band frequency by decreasing n_freq.")
        logger.error(msg)

    # Yes, this is absolutly hidious...
    # Sometimes when for example epochs_tmax = 5s,
    # the last element of np.arange is 4.99999, so we have to delete it...
    if np.isclose(centered_w_times_not_adjusted[-1], cfg.epochs_tmax):
        centered_w_times_not_adjusted = centered_w_times_not_adjusted[:-1]

    return n_time_windows


class Tf:
    """Util class containing usefull infos about the time frequency windows."""

    def __init__(self, cfg):
        """Calculate the required time and frequency size."""
        freqs = np.linspace(cfg.time_frequency_freq_min,
                            cfg.time_frequency_freq_max,
                            cfg.n_freqs)
        # make freqs list of tuples
        freq_ranges = list(zip(freqs[:-1], freqs[1:]))
        n_freq_windows = cfg.n_freqs - 1

        n_time_windows = compute_n_time_windows(cfg)
        # n_times is not really used in the rest of the code.
        # Same case for n_freqs.
        n_times = n_time_windows + 1

        msg = (f"We will split the epoch time interval"
               f" into {n_time_windows} windows.")
        logger.info(gen_log_message(msg, step=3))
        times = np.linspace(
            cfg.epochs_tmin, cfg.epochs_tmax, n_time_windows + 1)
        time_ranges = list(zip(times[:-1], times[1:]))
        centered_w_times = (times[1:] + times[:-1])/2

        # Recap:
        assert n_time_windows == n_times - 1
        assert len(time_ranges) == n_time_windows
        assert n_freq_windows == cfg.n_freqs - 1
        assert len(freq_ranges) == n_freq_windows

        self.freqs = freqs
        self.freq_ranges = freq_ranges
        self.times = times
        self.time_ranges = time_ranges
        self.centered_w_times = centered_w_times
        self.n_time_windows = n_time_windows
        self.n_freq_windows = n_freq_windows

        # TODO: Check that each subject contains the 2 conditions.
        # in order to avoid bad naws after 1 hour of computation ?


def prepare_labels(*, epochs: BaseEpochs, cfg) -> np.ndarray:
    """Return the projection of the events_id on a boolean vector.

    This projection is usefull in the case of hierarchical events:
    we project the different events contain in one condition into
    just one label.

    Returns:
    --------
    A boolean numpy array containing the labels.
    """
    tf_conditions = cfg.time_frequency_conditions
    assert len(tf_conditions) == 2

    # TODO: this create sometime a critical error sometime
    # but difficult to debug
    # because there is not the subject number written in the error.
    # "A critical error occurred. The error message was:
    # 'Event "3_item" is not in Epochs.
    # Event_ids must be one of "1_item/medium,
    # 1_item/short" The epochs.metadata Pandas query did not yield any
    # results: invalid syntax'"
    epochs_cond_0 = epochs[tf_conditions[0]]
    event_id_condition_0 = set(epochs_cond_0.events[:, 2])
    epochs_cond_1 = epochs[tf_conditions[1]]
    event_id_condition_1 = set(epochs_cond_1.events[:, 2])

    y = epochs.events[:, 2].copy()
    for i in range(len(y)):
        if y[i] in event_id_condition_0:
            y[i] = 0
        elif y[i] in event_id_condition_1:
            y[i] = 1
        else:
            msg = (f"Event_id {y[i]} is not contained in "
                   f"{tf_conditions[0]}'s set {event_id_condition_0}  nor in "
                   f"{tf_conditions[1]}'s set {event_id_condition_1}.")
            # TODO: Add subject number in this error?
            logger.warning(msg)
    return y


def prepare_epochs_and_y(
    *,
    epochs: BaseEpochs,
    cfg,
    fmin: float,
    fmax: float
) -> Tuple[BaseEpochs, np.ndarray]:
    """Band-pass and clean the epochs and prepare labels.

    Returns:
    --------
    epochs_filter, y
    """
    # Prepare epoch_filter
    epochs_filter = epochs.copy()
    epochs_filter.decimate(decim=cfg.decim)
    epochs_filter.filter(fmin, fmax, n_jobs=1, fir_design='firwin',
                         skip_by_annotation='edge')
    epochs_filter.drop_bad()

    epochs_filter.pick_types(  # TODO: Adapt
        meg=True, eeg=False, stim=False, eog=False,
        exclude='bads')

    # prepare labels
    y = prepare_labels(epochs=epochs_filter, cfg=cfg)

    return epochs_filter, y


def plot_frequency_decoding(
    *,
    freqs: np.ndarray,
    freq_scores: np.ndarray,
    conf_int: np.ndarray,
    subject: str,
    pth: Pth
) -> Figure:
    """Plot and save the frequencies results.

    Show and save the roc-auc score in a 1D histogram for
    each frequency bin.

    Parameters:
    -----------
    freqs
        The frequencies bins.
    freq_scores
        The roc-auc scores for each frequency bin.
    freq_scores_std
        The std of the cross-validation roc-auc scores for each frequency bin.
    subject
        name of the subject.
    pth
        Pth class.

    Returns:
    -------
    Histogramm with frequency bins.
    For the average subject, we also plot the std.
    """
    plt.close()
    fig, ax = plt.subplots()

    yerr = conf_int if len(subject) > 1 else None

    # The confidence intervals do not have any sense for individual subjects
    # yerr = freq_scores_std if subject == "avg" else None
    ax.bar(x=freqs[:-1], height=freq_scores, yerr=yerr,
           width=np.diff(freqs)[0],
           align='edge', edgecolor='black')

    # fixing the overlapping labels
    round_label = np.around(freqs)
    round_label = round_label.astype(int)
    ax.set_xticks(ticks=freqs)
    ax.set_xticklabels(round_label)

    ax.set_ylim([0, 1])

    ax.axhline(chance,
               color='k', linestyle='--',
               label='chance level')
    ax.legend()
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Decoding Scores')
    ax.set_title('Frequency Decoding Scores - 95% CI')

    return fig


def plot_time_frequency_decoding(
    *,
    freqs: np.ndarray,
    tf_scores: np.ndarray,
    sfreq: float,
    centered_w_times: np.ndarray,
    pth: Pth,
    subject: str
) -> Figure:
    """Plot and save the time-frequencies results.

    Parameters:
    -----------
    freqs
        The frequencies bins.
    tf_scores
        The roc-auc scores for each time-frequency bin.
    sfreq
        Sampling frequency # TODO: check if no border effect with decimate.
    centered_w_times
        List of times indicating the center of each time windows.
    subject
        name of the subject.

    Returns:
    -------
    The roc-auc score in a 2D map for each time-frequency bin.
    """
    plt.close()
    if np.isnan(tf_scores).any():
        msg = ("There is at least one nan value in one of "
               "the time-frequencies bins.")
        logger.info(gen_log_message(message=msg, step=3,
                                    subject=subject))
    tf_scores_ = np.nan_to_num(tf_scores, nan=chance)
    av_tfr = AverageTFR(create_info(['freq'], sfreq),
                        # newaxis linked with the [0] in plot
                        tf_scores_[np.newaxis, :],
                        centered_w_times, freqs[1:], 1)

    # Centered color map around the chance level.
    max_abs_v = np.max(np.abs(tf_scores_ - chance))
    figs = av_tfr.plot(
        [0],  # [0] We do not have multiple channels here.
        vmax=chance + max_abs_v,
        vmin=chance - max_abs_v,
        title="Time-Frequency Decoding ROC-AUC Scores",
    )
    return figs[0]


@failsafe_run(on_error=on_error)
def one_subject_decoding(
    *,
    cfg,
    tf: Tf,
    pth: Pth,
    subject: str,
    report: Report
) -> None:
    """Run one subject.

    There are two steps in this function:
    1. The frequency analysis.
    2. The time-frequency analysis.

    For each bins of those plot, we train a classifier to discriminate
    the two conditions.
    Then, we plot the roc-auc of the classifier.

    # TODO : profile this function

    Returns
    -------
    None. We just save the plots in the report
    and the numpy results in memory.
    """
    msg = f"Running decoding for subject {subject}..."
    logger.info(gen_log_message(msg, step=3, subject=subject))

    # Extract information from the raw file
    epochs = read_epochs(pth.file(subject=subject,
                                  session=config.get_sessions(),
                                  cfg=cfg))
    sfreq = epochs.info['sfreq']

    # Assemble the classifier using scikit-learn pipeline
    clf = make_pipeline(CSP(n_components=4, reg=cfg.reg,
                            log=True, norm_trace=False),
                        LinearDiscriminantAnalysis())

    cv = StratifiedKFold(n_splits=cfg.decoding_n_splits,
                         shuffle=True, random_state=42)

    ##########################################################################
    # 1. Loop through frequencies, apply classifier and save scores

    # init scores
    freq_scores = np.zeros((tf.n_freq_windows,))
    freq_scores_std = np.zeros((tf.n_freq_windows,))

    # Loop through each frequency range of interest
    for freq, (fmin, fmax) in tqdm(
            enumerate(tf.freq_ranges),
            total=tf.n_freq_windows,
            desc=f'subject {subject} - frequency loop'):

        epochs_filter, y = prepare_epochs_and_y(
            epochs=epochs, fmin=fmin, fmax=fmax, cfg=cfg)

        X = epochs_filter.get_data()

        # Save mean scores over folds
        cv_scores = cross_val_score(estimator=clf, X=X, y=y,
                                    scoring='roc_auc', cv=cv,
                                    n_jobs=1)

        freq_scores_std[freq] = np.std(cv_scores, axis=0)
        freq_scores[freq] = np.mean(cv_scores, axis=0)

    np.save(file=pth.freq_scores(subject), arr=freq_scores)
    np.save(file=pth.freq_scores_std(subject), arr=freq_scores_std)
    fig = plot_frequency_decoding(
        freqs=tf.freqs,
        freq_scores=freq_scores,
        conf_int=freq_scores_std,
        pth=pth,
        subject=subject)

    report.add_figs_to_section(  # TODO: That does not work
        fig,
        section="Frequency roc-auc decoding",
        captions=f'sub-{subject}')

    ######################################################################
    # 2. Loop through frequencies and time

    # init scores
    tf_scores = np.zeros((tf.n_freq_windows, tf.n_time_windows))

    # Loop through each frequency range of interest
    for freq, (fmin, fmax) in tqdm(
            enumerate(tf.freq_ranges),
            total=tf.n_freq_windows,
            desc=f'subject {subject} - time-frequency loop'):

        epochs_filter, y = prepare_epochs_and_y(
            epochs=epochs, fmin=fmin, fmax=fmax, cfg=cfg)

        # Roll covariance, csp and lda over time
        for t, (w_tmin, w_tmax) in enumerate(tf.time_ranges):

            # Originally the window size varied accross frequencies...
            # But this means also that there is some mutual information between
            # 2 different pixels in the map.
            # So the simple way to deal with this is just to fix
            # the windows size for all frequencies.

            # Crop data into time-window of interest
            X = epochs_filter.copy().crop(w_tmin, w_tmax).get_data()

            # Save mean scores over folds for each frequency and time window
            cv_scores = cross_val_score(estimator=clf,
                                        X=X, y=y,
                                        scoring='roc_auc',
                                        cv=cv,
                                        n_jobs=1)

            # TODO: add CSP plots

    np.save(file=pth.tf_scores(subject), arr=tf_scores)
    fig = plot_time_frequency_decoding(
        freqs=tf.freqs, tf_scores=tf_scores, sfreq=sfreq, pth=pth,
        centered_w_times=tf.centered_w_times, subject=subject)
    report.add_figs_to_section(
        fig,
        section="Time - frequency decoding",
        captions=f'sub-{subject}')

    msg = f"Decoding for subject {subject} finished succesfully."
    logger.info(gen_log_message(message=msg, subject=subject, step=3))


def load_and_average(
    path: Callable[[str], str],
    subjects: List[str],
    average: bool = True
) -> np.ndarray:
    """Load and average a np.array.

    Parameters:
    -----------
    path
        function of the subject returning the path of the numpy array.
    average
        if True, returns average along the subject dimension.

    Returns:
    --------
    The loaded array.

    Warning:
    --------
    Gives the list of files containing NaN values.
    """
    res = np.zeros_like(np.load(path(subjects[0])))
    shape = [len(subjects)]+list(res.shape)
    res = np.zeros(shape)
    for i, sub in enumerate(subjects):
        try:
            arr = np.load(path(sub))
        except FileNotFoundError:
            # Not exactly what to do with confidence intervals...
            # Maybe 0 is not here also
            arr = np.empty_like(res[0])
            arr.fill(np.NaN)
        if np.isnan(arr).any():
            msg = f"NaN values were found in {path(sub)}"
            logger.warning(gen_log_message(
                message=msg, subject=sub, step=3))
        res[i] = arr
    if average:
        return np.nanmean(res, axis=0)
    else:
        return res


def plot_axis_time_frequency_statistics(
    *,
    ax: plt.Axes,
    array: np.ndarray,
    value_type: Literal['p', 't'],
    subjects: List[str],
    cfg,
    tf: Tf
) -> None:
    """Plot one 2D axis containing decoding statistics.

    Parameters:
    -----------
    ax: pyplot axis.
    array : np.ndarray
    value_type : either "p" or "t"
    cfg : BunchConst

    Returns:
    --------
    None. inplace modification of the axis.
    """
    ax.set_title(f"{value_type}-value")
    array = np.maximum(array, 1e-7) if value_type == "p" else array
    array = np.reshape(array, (tf.n_freq_windows, tf.n_time_windows))
    array = -np.log10(array) if value_type == "p" else array

    # adaptative color plot ?
    lims = np.array([np.min(array), np.max(array)])

    img = ax.imshow(array, cmap='Reds', origin='lower',
                    vmin=lims[0], vmax=lims[1], aspect='auto',
                    extent=[np.min(tf.times), np.max(tf.times),
                            np.min(tf.freqs), np.max(tf.freqs)])

    ax.set_xlabel('time')
    ax.set_ylabel('frequencies')
    cbar = plt.colorbar(ax=ax, shrink=0.75, orientation='horizontal',
                        mappable=img, )
    cbar.set_ticks(lims)
    cbar.set_ticklabels([round(lim, 1) for lim in lims])
    cbar.ax.get_xaxis().set_label_coords(0.5, -0.3)
    if value_type == "p":
        cbar.set_label(r'$-\log_{10}(p)$')
    if value_type == "t":
        cbar.set_label(r't-value')


def plot_t_and_p_values(
    t_values: np.ndarray,
    p_values: np.ndarray,
    title: str,
    subjects: List[str],
    cfg,
    tf: Tf
) -> Figure:  # TODO : delete this function ?
    """Plot t-values and either (p-values or clusters).

    Parameters:
    -----------
    t-values
    p_values
    title
    cfg
    tf

    Returns
    -------
    A figure with two subplot: t-values and p-values.
    """
    plt.close()
    fig = plt.figure(figsize=(10, 5))
    axes = [fig.add_subplot(121), fig.add_subplot(122)]

    plot_axis_time_frequency_statistics(
        ax=axes[0], array=t_values, cfg=cfg,
        subjects=subjects, value_type="t", tf=tf)
    plot_axis_time_frequency_statistics(
        ax=axes[1], array=p_values, cfg=cfg,
        subjects=subjects,  value_type="p", tf=tf)
    plt.tight_layout()
    fig.suptitle(title)
    return fig


@failsafe_run(on_error=on_error)
def group_analysis(
    subjects: List[str],
    cfg,
    pth: Pth,
    tf: Tf,
    report: Report,
) -> None:
    """Group analysis.

    1. Average roc-auc scores:
        - frequency 1D histogram
        - time-frequency 2D color-map
    2. Perform statistical tests
        - plot t-values and p-values
        - performs classic cluster permutation test

    Returns
    -------
    None. Plots are saved in memory.
    """
    if len(subjects) == 0:
        msg = "We cannot run a group analysis with just one subject."
        logger.critical(gen_log_message(msg, step=3))

    msg = "Running group analysis..."
    logger.info(gen_log_message(msg, step=3))

    ######################################################################
    # 1. Average roc-auc scores across subjects

    # just to obtain sfreq
    epochs = read_epochs(pth.file(subject=subjects[0],
                                  session=config.get_sessions(),
                                  cfg=cfg))
    sfreq = epochs.info['sfreq']

    # Average Frequency analysis
    all_freq_scores = load_and_average(pth.freq_scores, subjects=subjects)

    # Calculating the 95% confidence intervals
    # TODO: The confidence intervals indicated on the individual
    # plot are not quite valid because
    # they use only the std of the cross-validation scores.
    # We should also divide by the square root.
    all_freq_scores_std = load_and_average(
        pth.freq_scores, subjects=subjects, average=False)
    std_ = np.nanstd(all_freq_scores_std, axis=0, ddof=1)
    conf_int = 1.96*std_ / np.sqrt(all_freq_scores_std.shape[0])

    fig = plot_frequency_decoding(
        freqs=tf.freqs, freq_scores=all_freq_scores, pth=pth,
        conf_int=conf_int, subject="all")
    report.add_figs_to_section(
        fig,
        section="Frequency decoding",
        captions='sub-all')

    # Average time-Frequency analysis
    all_tf_scores = load_and_average(pth.tf_scores, subjects=subjects)

    fig = plot_time_frequency_decoding(
        freqs=tf.freqs, tf_scores=all_tf_scores, sfreq=sfreq, pth=pth,
        centered_w_times=tf.centered_w_times, subject="all")
    report.add_figs_to_section(
        fig,
        section="Time - frequency decoding",
        captions='sub-all')

    ######################################################################
    # 2. Statistical tests

    # Reshape data to what is equivalent to (n_samples, n_space, n_time)
    X = load_and_average(pth.tf_scores, subjects=subjects, average=False)
    X = X - chance

    # TODO: Perform test without filling nan values.
    # The permutation test do not like nan values.
    # So I fill in the nan values with the average of each column.
    # nanmean accentuate the effect? A little bit augmenting
    # artificially the number of subjects.
    col_mean = np.nanmean(X, axis=0)
    # Find indices that you need to replace
    inds = np.where(np.isnan(X))
    # Place column means in the indices. Align the arrays using take
    X[inds] = np.take(col_mean, inds[1])

    # Analyse with cluster permutation statistics
    titles = ['Without clustering']
    out = stats.ttest_1samp(X, 0, axis=0)
    ts: List[np.ndarray] = [np.array(out[0])]  # statistics
    ps: List[np.ndarray] = [np.array(out[1])]  # pvalues

    mccs = [False]  # these are not multiple-comparisons corrected

    titles.append('Clustering')
    # Compute threshold from t distribution (this is also the default)
    threshold = stats.distributions.t.ppf(1 - cfg.alpha_t_test,
                                          len(subjects) - 1)
    t_clust, clusters, p_values, H0 = permutation_cluster_1samp_test(
        X, n_jobs=1,
        threshold=threshold,
        adjacency=None,  # a regular lattice adjacency is assumed
        n_permutations=cfg.n_permutations, out_type='mask')

    # TODO: plot H0 ?
    msg = "Permutations performed succesfully"
    logger.info(gen_log_message(msg, step=3))
    # Put the cluster data in a viewable format
    p_clust = np.ones((tf.n_freq_windows, tf.n_time_windows))
    for cl, p in zip(clusters, p_values):
        p_clust[cl] = p
    msg = (f"We found {len(p_values)} clusters "
           f"each one with a p-value of {p_values}.")
    logger.info(gen_log_message(msg, step=3))

    if np.min(p_values) > cfg.alpha:
        msg = ("The results are not significative. "
               "Try increasing the number of subjects.")
        logger.info(gen_log_message(msg, step=3))
    else:
        msg = (f"Congrats, the results seem significative. At least one of "
               f"your cluster has a significant p-value "
               f"at the level {cfg.alpha}")
        logger.info(gen_log_message(msg, step=3))

    ts.append(t_clust)
    ps.append(p_clust)
    mccs.append(True)
    for i in range(2):
        fig = plot_t_and_p_values(
            t_values=ts[i], p_values=ps[i], title=titles[i],
            subjects=subjects, cfg=cfg, tf=tf)

        cluster = "with" if i else "without"
        report.add_figs_to_section(
            fig,
            section=f"Time - frequency statistics - {cluster} cluster",
            captions='sub-all')
    msg = "Group analysis statistic analysis finished."
    logger.info(gen_log_message(msg, step=3))


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> BunchConst:
    cfg = BunchConst(
        datatype=config.get_datatype(),  # TODO: 'meg' use
        deriv_root=config.get_deriv_root(),  # TODO
        time_frequency_conditions=config.time_frequency_conditions,
        ch_types=config.ch_types,  # TODO
        time_frequency_freq_min=config.time_frequency_freq_min,
        time_frequency_freq_max=config.time_frequency_freq_max,
        decim=config.decim,
        decoding_n_splits=config.decoding_n_splits,
        epochs_tmin=config.epochs_tmin,
        epochs_tmax=config.epochs_tmax,
        task=config.task,
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        n_freqs=config.n_freqs,
        n_cycles=config.n_cycles,
        reg=config.reg,
        alpha=config.alpha,
        alpha_t_test=config.alpha_t_test,
        n_permutations=config.n_permutations
    )
    return cfg


def main():
    """Run all subjects decoding in parallel."""
    msg = 'Running Step 3: Time-frequency decoding'
    logger.info(gen_log_message(message=msg, step=3))

    cfg = get_config()

    # Calculate the appropriate time and frequency windows size
    tf = Tf(cfg)

    # Compute the paths
    pth = Pth(cfg=cfg)

    report = Report(title="csp-permutations", verbose=False)

    # Usefull for debuging:
    # [one_subject_decoding(
    #     cfg=cfg, tf=tf, pth=pth, subject=subject, report=report)
    #     for subject in config.get_subjects()]
    parallel, run_func, _ = parallel_func(one_subject_decoding, n_jobs=N_JOBS)
    parallel(run_func(cfg=cfg, tf=tf, pth=pth, subject=subject, report=report)
             for subject in config.get_subjects())

    # Once every subject has been calculated,
    # the group_analysis is very fast to compute.
    group_analysis(subjects=config.get_subjects(),
                   cfg=cfg, pth=pth, tf=tf,
                   report=report)

    report.save(pth.report_fname, overwrite=True,
                open_browser=config.interactive)

    msg = f"Report {pth.report_fname} saved in the average subject folder"
    logger.info(gen_log_message(message=msg, step=3))

    msg = 'Completed Step 3: Time-frequency decoding'
    logger.info(gen_log_message(message=msg, step=3))


if __name__ == '__main__':
    main()
