# Design

What the system is and why: scope, architecture, the key decisions with the alternatives I rejected, how results are tested, and the assumptions and limitations. How each stage actually went, including the problems I found and fixed, is in [DEVLOG.md](DEVLOG.md).

---

## Problem and scope

To measure detection completeness: the fraction of planets recovered (true positives over all positives) as a function of signal strength, for six pipelines on identical injected data:

- Box Least Squares alone
- Transit Least Squares alone
- Box Least Squares followed by a convolutional vetting network
- Transit Least Squares followed by a convolutional vetting network
- Box Least Squares followed by a random forest vetting
- Transit Least Squares followed by a random forest vetting

Plus two *oracle* arms (oracle→CNN and oracle→RF), where the detector is replaced by the true injected period and mid-transit time. They are not competitors but upper bounds on the vetting stage (see Points for discussion).

I'm choosing to test 2 classical periodograms and 2 AI models (with the initial detection done by the periodograms, see next section).
It provides a sufficient comparison to judge the effectiveness of the vetting stage. <br> This project is not about discovering new planets or beating standard models like Robovetter, but rather a practical experiment to answer the posed question. <br> *Out of scope*: multi-planet systems, Gaussian-process detrending, and Bayesian transit fitting. Their mathematics sits beyond what I can properly understand inside the project's time budget.

## An important distinction: detection versus vetting

These two are different tasks, and confusing them could lead to an invalid headline result.

- **Detection** takes a light curve and returns a period found. BLS and TLS do this by
  searching a grid of trial periods.
- **Vetting** takes a signal at a *known* period and returns a probability that it is a
  planet. The CNN and RF will do this, as does NASA's Robovetter.

Therefore, a direct "CNN versus BLS" comparison is meaningless: the CNN cannot run
without a period, which is what BLS supplies.

## Architecture

*Module map* (as of Stage 7):

```
transit_lab/                 the package: library code only, no I/O or data
  transit_model.py           transit light curve: uniform source and quadratic limb darkening (Stage 3)
  vetting.py                 views -> RF features / CNN tensors; RF and dual-view CNN; training, scoring, metrics (Stage 7)
scripts/                     entry points: file I/O and command-line options, logic lives in the package
  compare_vetters.py         Paired bootstrap: frozen CNN vs frozen RF on AstroNet's test split
  convert_tfrecords.py       AstroNet TFRecords -> astronet_data/{train,val,test}.npz, with checks and the star-disjoint mask
  results_cnn.py             Evaluate the performance of a saved CNN from a .pt file. Report vetting metrics.
  train_rf.py                RF per seed, validation metrics (mean ± std over seeds)
  train_cnn.py               CNN per seed, early stopping on val, saves intomodels/cnn_seed*.pt
tests/
  test_transit_model.py      transit model against batman, contact points, symmetries
  test_vetting.py            classifiers and metrics on small synthetic data (no downloads, runs in CI)
astronet_data/               converted .npz views and labels (committed; the original TFRecords are not)
models/                      CNN checkpoints
paper/report.tex             technical report: methods and the transit-model derivations
```

*Data flow*:

- **Stage 7, done**: AstroNet TFRecords → `convert_tfrecords.py` → `.npz` → `vetting.py` (RF / CNN) → planet scores → AUC, PR-AUC.
- **Full pipeline, planned (Stages 2, 4–6, 8–10)**: Kepler light curve → conditioning → inject a synthetic transit (`transit_model.py`) → detrending → BLS / TLS (period, mid-transit time) → global and local views → RF / CNN score → threshold from the control run → recovered or not → completeness curve.

*Boundaries*. The package holds pure, tested functions; scripts only read files, parse options, and call the package. The vetting code takes the same `.npz` columns whatever the source, so the views my own Stage 6 code produces from injected light curves go through exactly the same scoring path as AstroNet's.

## Key decisions

Each row is a decision with the alternative I rejected. The full story, where there is one, is in the log.

| Decision | Alternative rejected | Why | Log |
|---|---|---|---|
| Kepler data only | TESS | Kepler observed one field for 4 years at a fixed 29.4-min cadence; detectability grows with the number of transits, so TESS's ~27-day sectors would lose the long-period end. The labelled training set is also Kepler. | |
| Circular orbits | Eccentric orbits | The separation formula I derived assumes a circular orbit; eccentricity (e, ω) would add two more injection parameters. | |
| Quadratic limb darkening via a numerical ring integral, N = 4000 rings | Closed-form elliptic integrals (Mandel & Agol 2002) | Derivable and understandable within the budget; agrees with `batman` to within 10⁻⁶. | [Stage 3](DEVLOG.md#stage-3) |
| Train the classifiers on AstroNet's pre-computed views | Generating training views from raw light curves myself | Avoids a long preprocessing step before any model exists. My own view code (Stage 6) will be checked against these exact vectors. | [Stage 7a](DEVLOG.md#stage-7a-preparing-data-for-vetting) |
| Keep AstroNet's released split (by TCE) | Re-splitting by star | Results stay comparable with Shallue & Vanderburg (2018); the star leak is measured with a star-disjoint test subset instead. | [Stage 7a](DEVLOG.md#stage-7a-preparing-data-for-vetting) |
| Commit the converted `.npz` files (~109 MB) | Download from AstroNet's Google Drive link; GitHub release asset | Reproduction must not depend on a link in an archived repository. The files are frozen: a changed converter does not get its outputs recommitted. | |
| Random Forest as a full pipeline arm, on the same views as the CNN | RF as an offline baseline only; RF on hand-engineered features | Identical inputs isolate the model class, in both the AUC comparison and the completeness curves. | [Stage 7b](DEVLOG.md#stage-7b-training-the-rf-using-scikit-learn) |
| Glorot-uniform weights, zero biases | PyTorch's default initialisation | Matches TensorFlow's defaults used by AstroNet; fixed a slow start (val AUC stuck at about 0.77 for 5 epochs). | [Stage 7c](DEVLOG.md#stage-7c-building-a-cnn) |
| Up to 300 epochs, early stopping on val AUC (patience 15) | The paper's fixed 50 epochs | My PyTorch port was still improving at epoch 50; the epoch is chosen on val, not on test. | [Stage 7c](DEVLOG.md#stage-7c-building-a-cnn) |
| One CNN per seed; seed 0 frozen for test and later stages | Averaging 10 models, as the paper does | A tenth of the training time, and the same single model scores the injections later. | [Stage 7c](DEVLOG.md#stage-7c-building-a-cnn) |
| Save the CNN as a checkpoint; retrain the RF from its seed | Saving both, or retraining both | The CNN takes minutes and isn't bit-for-bit reproducible on MPS; the RF retrains identically in 12 s, and a saved forest is large and tied to one scikit-learn version. | [Stage 7d](DEVLOG.md#stage-7d-comparing-the-test-results-for-frozen-rf-and-cnn-and-performing-paired-bootstrapping) |
| Inject before detrending | Detrending first, then injecting | Detrending can partly remove a real transit; injecting afterwards avoids the damage a real transit curve would have taken and gives optimistic completeness. | |

## Testing and results. Two types of tests:

*Headline* — end-to-end pipeline completeness. All six main pipelines take a raw light
curve and return yes/no, so they are directly comparable. This answers the question that
actually matters: does adding a learned vetting stage recover more real planets at a
fixed false-alarm rate?

*Supporting* — algorithm validations and stage-isolated sensitivity. For derived and implemented algorithms,
I will run them against the reference implementations to validate them. For vetting, the CNN and RF recovery as a function of
signal-to-noise, given the true period and mid-transit time (the oracle arms). This is the more scientifically informative
number but is generous to the classifier, since in deployment the period arrives from an
imperfect detector like BLS. It will be reported as an upper bound.

*Fair comparison.* Every pipeline gets its own threshold from a control run (item 8), set so that all of them have the same false-alarm rate. Raw detection statistics (e.g. the SDE of BLS and TLS) are defined differently, so comparing them directly would be meaningless; comparing recovery at a matched false-alarm rate is not.

*Classifier evaluation.* The test set is evaluated once, on frozen models, on the full set and on the star-disjoint subset. The CNN–RF difference is judged with a paired bootstrap over test TCEs.

---

## Assumptions and limitations

*Assumptions*
1. The DR24 Autovetter training labels (PC / AFP / NTP) are treated as ground truth. Shallue & Vanderburg found some mislabelled examples, so this is not fully accurate.
2. Synthetic transits are injected into real light curves, so the noise is genuine and only the signal is synthetic.
3. Circular orbits and Kepler long-cadence data only (see Key decisions).

*Limitations*
1. Star overlap across AstroNet's split: val and test metrics are likely optimistic. The star-disjoint subset bounds the effect but doesn't measure it, since it is smaller and its composition differs (35% planets against about 23%) ([Stage 7a](DEVLOG.md#stage-7a-preparing-data-for-vetting)).
2. Domain shift: the classifiers are trained on real TCE views and tested on synthetic injections. On top of that, all training TCEs passed Kepler's detection threshold (MES > 7.1), and only 8% of training planets have MES < 10 (Shallue & Vanderburg 2018). The faintest injections will be largely outside the training distribution.
3. The transit model is written in pure Python and evaluates one time stamp at a time; it must be vectorised before the injection–recovery runs ([Stage 3](DEVLOG.md#stage-3)).
4. The oracle arms are optimistic in two ways: perfect period recovery, and a false-alarm population that is not selected by a detector (Points for discussion).
5. The view-generation code (Stage 6) must reproduce AstroNet's conventions exactly, including the spline normalisation, or the classifiers see a different input distribution than they were trained on.

## Points for discussion

1. **AstroNet is a correctness reference, not an opponent.** I use its architecture, its pre-computed views, and its labels, so there's nothing to compete on. It plays the same role `batman` plays for my transit model and `astropy` for my BLS. The CNN's real opponent in the headline is the absence of a vetting stage.
2. **A strong baseline changes the question.** An untuned RF on raw bins reaches val AUC 0.97, so most of the benchmark is easy (obvious eclipsing binaries, noise). The models can only differ on the hard, faint cases, which a single AUC under-represents. *Expectation, written before the injection runs*: the CNN's advantage over the RF will concentrate at low S/N, because convolution gets evidence across neighbouring bins, so this local feature extraction becomes crucial at low S/N, when the data is not so clear (as RF might expect). It might still fail: both classifiers may collapse together at low S/N because such signals are out of their training distribution (Limitations, 2). But either outcome is a finding.
3. **Two kinds of uncertainty.** Seed variation measures how much the model changes; sampling error measures how much the score changes because the evaluation set is a finite sample. For both classifiers the sampling error is about 8–10 times larger, so comparisons are judged by a paired bootstrap, not by seed spread. A refinement: TCEs on the same star aren't independent, so resampling stars instead of TCEs (a cluster bootstrap) would give a more honest interval.
4. **Oracle arms are bounds, not competitors.** Oracle→CNN sits above BLS→CNN by construction; the interesting number is the gap, which is the cost of imperfect detection. On control light curves there is no true period, so the oracle arms are calibrated by folding at random periods drawn from the injection grid. Those false alarms are easier to reject than BLS's spurious peaks, which tend to look transit-like.
5. **Leakage reaches the completeness curves too.** If the stars used for injection were in the CNN's training set, the completeness curves inherit the same star leakage as the test metrics. Injection stars must be drawn from outside the classifiers' training stars.