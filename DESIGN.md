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