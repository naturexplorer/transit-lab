# transit-lab

**Does adding an learned vetting stage recover more planets than classical period search alone?**

This hands-on project is a natural follow-up of my theoretical EPQ answering the question
*"To what extent does AI assist in exoplanet detection using transit photometry?"*
Transit photometry discovers planets by looking for small periodic dips in a star's
brightness as a planet crosses in front of it. The classical approach searches for
periodic box-shaped dips, such as the Box Least Squares (BLS); the modern approach 
trains convolutional networks to recognise planet-like signals. Published work
mainly reports accuracy figures on fixed benchmarks. This project asks the more useful
question: **at what signal strength does each approach start to work, and where does each break down?**

The method I use is injection and recovery. Synthetic transits of known size and period are
injected into real Kepler light curves, then each pipeline is run to see whether it
finds them. The result is a *completeness curve*: recovery fraction against
signal-to-noise, with every pipeline calibrated to the same false-alarm rate so the
comparison is fair.

This practical investigation will involve:
1. writing tests and CI, pinning dependencies, and writing a design document to update later
2. conditioning the data: dropping bad cadences, NaN removal, per-quarter normalisation, and stitching the quarters together (detrending comes after injection, see 10)
3. implementing the transit model to inject simulated transit curves onto real Kepler light curves. Test: Validated against `batman`
4. implementing the BLS algorithm for transit detection. Test: Validated against `astropy`
5. borrowing the Transit Least Squares (TLS) algorithm as another data point for comparison
6. writing a pre-processing module producing AstroNet-convention global and local views from a light curve. Test: Checked against `astronet`
7. training a dual-view CNN and a Random Forest classifier on real Kepler TCEs for vetting, using identical views and splits so the comparison isolates model class. Test: held-out AUC and PR-AUC reports for both
8. performing a control run to identify the thresholds for all 8 pipelines using a fixed false-alarm rate to ensure fair comparison
9. injecting synthetic transits into conditioned light curves, detrending, and then applying the 8 pipelines below in the table.
10. comparing the results by plotting the *completeness curve*: recovery fraction against signal-to noise, period, and impact parameter (S/N is fixed for last two)
11. crafting a CLI that reproduces every figure in the written report for credibility
12. Gathering the real attained results and completing the README and written report

### Pipelines
| Detector | Vetter | Comment |
|---|---|---|
| BLS | - | classical baseline |
| TLS | - | classical baseline |
| BLS | CNN |  |
| TLS | CNN |  |
| BLS | RF |  |
| TLS | RF |  |
| oracle | CNN | upper bound |
| oracle | RF | upper bound |

## Headline result
Headline plot with three pipelines, not eight (for readability): BLS, BLS->CNN, BLS->RF
*Pending results.*