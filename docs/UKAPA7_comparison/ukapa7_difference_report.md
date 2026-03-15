# UKAPA7 Difference Distribution Report

This report compares two full-wire tension measurements for UKAPA7 layer G:

- JSON values: measured at the UK factory on December 10, 2023
- CSV values: measured at the Chicago factory on March 11, 2026

Residual definition: `Chicago - UK`.

## Source Artifacts

- Comparison CSV: `data/tension_summaries/tension_comparison_UKAPA7_G.csv`
- Per-wire residual plot: `data/tension_plots/tension_diff_per_wire_UKAPA7_G.png`
- Residual histogram plot: `data/tension_plots/tension_diff_distribution_UKAPA7_G.png`

## Physical Expectation

The expected physical change between the two measurement campaigns is a
uniformly negative and approximately constant residual, meaning the Chicago
values should tend to lie below the earlier UK values by roughly the same amount
across wires.

Both measurements are also noisy. Using the stated single-measurement noise
scale of about 0.25 N to 0.5 N, the expected residual scatter from measurement
noise alone is about 0.35 N to 0.71 N, assuming the two measurements are
independent. That combined-noise range is an inference from the stated
per-measurement uncertainty.

## Plots

### Per-Wire Residuals

![UKAPA7 per-wire residuals](../data/tension_plots/tension_diff_per_wire_UKAPA7_G.png){ width=100% }

### Residual Distributions

![UKAPA7 residual distributions](../data/tension_plots/tension_diff_distribution_UKAPA7_G.png){ width=100% }

## Summary

- Both sides are fully populated in the current comparison: 481 wires on side A
  and 481 wires on side B.
- The mean residual is negative on both sides, consistent with the expected sign
  of the physical change from UK to Chicago.
- Side A shows the stronger negative shift: mean residual `-0.3762 N`.
- Side B also shifts negative, but more weakly: mean residual `-0.1994 N`.
- The observed residual standard deviations are about `0.41 N` on both sides,
  which lies inside the inferred `0.35 N` to `0.71 N` band expected from the
  stated measurement noise.
- The residuals are not uniformly negative on every wire. Side A is negative on
  `84.4%` of wires and side B is negative on `68.0%` of wires.

## Numerical Statistics

| Side | Count | Mean (N) | Median (N) | Std Dev (N) | Mean Abs (N) | Max Abs (N) | Correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 481 | -0.3762 | -0.4269 | 0.4103 | 0.4843 | 1.6307 | 0.2267 |
| B | 481 | -0.1994 | -0.1976 | 0.4188 | 0.3712 | 1.4601 | 0.2319 |

### Quantiles

| Side | Min (N) | 5% | 25% | 75% | 95% | Max (N) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | -1.6307 | -0.9978 | -0.5977 | -0.2002 | 0.4244 | 0.9165 |
| B | -1.4601 | -0.8971 | -0.4831 | 0.0645 | 0.5196 | 1.1245 |

### Residual Sign

| Side | Negative | Positive |
| --- | ---: | ---: |
| A | 84.4% | 15.6% |
| B | 68.0% | 32.0% |

## Largest Absolute Residuals

### Side A

| Wire | UK JSON (N) | Chicago CSV (N) | Residual (N) | Abs Residual (N) |
| --- | ---: | ---: | ---: | ---: |
| 412 | 6.9000 | 5.2693 | -1.6307 | 1.6307 |
| 479 | 6.4900 | 5.0000 | -1.4900 | 1.4900 |
| 430 | 6.7200 | 5.2911 | -1.4289 | 1.4289 |
| 481 | 6.3800 | 5.0000 | -1.3800 | 1.3800 |
| 400 | 6.3800 | 5.0593 | -1.3207 | 1.3207 |

### Side B

| Wire | UK JSON (N) | Chicago CSV (N) | Residual (N) | Abs Residual (N) |
| --- | ---: | ---: | ---: | ---: |
| 287 | 5.9700 | 4.5099 | -1.4601 | 1.4601 |
| 185 | 5.7300 | 4.4422 | -1.2878 | 1.2878 |
| 64 | 6.6900 | 5.5041 | -1.1859 | 1.1859 |
| 461 | 6.8500 | 5.7248 | -1.1252 | 1.1252 |
| 177 | 5.8800 | 7.0045 | 1.1245 | 1.1245 |

## Interpretation

- The negative means on both sides are directionally consistent with the
  expected physical change from the UK measurement on December 10, 2023 to the
  Chicago measurement on March 11, 2026.
- The observed residual widths, about `0.41 N`, are compatible with the stated
  measurement noise model once both campaigns contribute to the residual.
- If the physical change were perfectly constant wire-to-wire, the residual line
  plots would fluctuate around a single flat level. Instead, the residuals show
  noticeable wire-to-wire structure on top of the negative offset, especially on
  side A.
- Side A looks more consistent with a substantial negative shift: its median is
  `-0.4269 N` and only `15.6%` of wires are positive.
- Side B still has a negative mean and median, but it is less uniformly
  negative, with `32.0%` positive residuals. That suggests either a smaller
  physical shift on side B, larger effective noise, or additional side-dependent
  variation beyond a single constant offset.

## Bottom Line

The UK-to-Chicago comparison is consistent with the expected negative direction
of change on both sides. The residual spread is also compatible with the stated
measurement noise scale. However, the residuals are not perfectly constant
wire-to-wire, and side B in particular is only moderately biased negative rather
than uniformly so.
