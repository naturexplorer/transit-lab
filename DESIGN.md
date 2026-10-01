# Design

Architecture and the reasoning behind it. The README had 12 points. When I make a decision or change the plan for that item, I will record it under that number in that list. Other important choices and design decisions will be labelled by name describing the category it is about.

---

## Problem and scope

To measure detection completeness: the fraction of planets recovered (true positives over all positives) as a function of signal strength, for six pipelines on identical injected data:

- Box Least Squares alone
- Transit Least Squares alone
- Box Least Squares followed by a convolutional vetting network
- Transit Least Squares followed by a convolutional vetting network
- Box Least Squares followed by a random forest vetting
- Transit Least Squares followed by a random forest vetting

I'm choosing to test 2 classical periograms and 2 AI models (with the initial detection done by the periograms, see next point).
It provides a sufficient comparison to judge the effectiveness of the vetting stage.

This project is not about dicovering new planets or beating standard models like Robovetter, but rather a practical experiment to answer the posed question.

So far limitations I accept:
1. no eccentric orbits, only circlular (it messes up speed and the transit model I plan to derive will not work).
2. only Kepler data is used, no TESS. Kepler has longer data periods, fixed cadence, and is more consistent with labelling.

## An important distinction: detection versus vetting

These two are different tasks, and confusing them could lead to an invalid headline result.

- **Detection** takes a light curve and returns a period found. BLS and TLS do this by
  searching a grid of trial periods.
- **Vetting** takes a signal at a *known* period and returns a probability that it is a
  planet. The CNN and RF will do this, as does NASA's Robovetter.

Therefore, a direct "CNN versus BLS" comparison meaningless: the CNN cannot run
without a period, which BLS is what supplies.

## Testing and results. Two types of tests:

*Headline* — end-to-end pipeline completeness. All six main pipelines take a raw light
curve and return yes/no, so they are directly comparable. This answers the question that
actually matters: does adding a learned vetting stage recover more real planets at a
fixed false-alarm rate?

*Supporting* — algorithm validations and stage-isolated sensitivity. For derived and implemented algorithms,
I will run them against the true models to validate them. For vetting, the CNN and RF recovery as a function of
signal-to-noise, given the true period and mid-transit time. This is the more scientifically informative
number but is generous to the classifier, since in deployment the period arrives from an
imperfect detector like BLS. It will be reported as an upper bound.

## Stage 1

I set up the project configuration so "import transit_lab" is possible for testing and other purposes.
Additionally, CI is set up, and tests will be added as I progress, before the stage I am embarking on.

## Stage 3

This stage came before Stage 2 as I decided to do the astronomical geometry first and implement the model. That is fine as Stage 2 is simply data cleaning and preparation.
I am only implementing uniform source and quadratic limb darkening cases as they are mathematically accessible for me, and they give the most understanding and practical usability for future injections.

Tests at the contact points of star and planet caught a rounding error which produces math.acos argument slightly outside [-1, 1]. I fix it with clamping. Tests pass.

## Stage 7a: preparing data for vetting

I chose to do Stage 7 before Stage 2 because RF and CNN will trained on provided data (pre-computed views), and conditioning of the raw data will be used for later stages, when injection happens. So order is fine.

AstroNet provides pre-computed views in TFRecords, which I have to unpack. I convert them into .npz for easy use.

Writing the conversion script revealed that there are quite a lot of shared stars across training, validation, and test data. AstroNet's released split is random by TCE, not by star: 851 of 1499 test stars also appear in train (57%). Since TCEs on the same star share noise and tend to share labels (stars with one planet often host more), this likely inflates test metrics. I want to keep the released split for comparability with Shallue & Vanderburg (2018), and quantify the leak instead. convert_tfrecords.py flags a star-disjoint test subset ([N] TCEs, [P] planets, [S] stars) containing only stars absent from train and val. Every metric will be reported on both the full and star-disjoint test sets.

## Stage 7b: training the RF using scikit-learn

*Configuration of RF*. sklearn.ensemble.RandomForestClassifier: 500 trees, all other hyperparameters at defaults (√2202 ≈ 47 bins considered per split, unlimited depth). Input: global and local views concatenated into a 2202-dim vector, the same inputs as the CNN, so the comparison isolates the model class. Untuned by design: no decision has yet been made using val. Trains in about 12 s on Macbook Pro M5.

Validation results (AstroNet val split, 1574 TCEs, 355 planets; 5 seeds, mean ± sample std):

| Metric | RF	| Random ranking
|---|---|---|
| ROC AUC | 0.9706 ± 0.0005 | 0.5 |
| PR-AUC | 0.9062 ± 0.0006 | 0.226 |

*Uncertainty*. Using the standard Hanley–McNeil formula for standard error (SE) of an AUC with positive and negative counts 355 and 1219 respectively, the SE is about 0.006. This is >10 times the Seed variation (±0.0005). Therefore, it suggests that the sampling error carries much more uncertainty for the model than the seed variations. 

*Interpretation*. This untuned RF on raw bins already exceeds the 0.95 AUC gate set for the CNN, so a significant share of the benchmark is separable without learned convolutional features. The published single-model CNN (AUC ≈ 0.989) makes around a third as many ranking errors (1 − AUC of 0.011 against 0.029).

*Limitations*. Val shares 849 stars with train (the released split is by TCE, not kepid), so these figures are probably optimistic. Test (--test) has not been evaluated: I reserve it for the frozen models, with final results reported on both the full set and the unseen-star subset.

## Stage 7c: building a CNN