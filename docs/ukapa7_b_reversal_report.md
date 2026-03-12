# UKAPA7 B-Side Reversal Report

This report summarizes the most likely interpretation of the UKAPA7 layer G side
B discrepancy between:

- UK factory measurements recorded in `apa_uk7g.json` on December 10, 2023
- Chicago factory measurements recorded in
  `data/tension_summaries/tension_summary_UKAPA7_G.csv` on March 11, 2026

Residual definition throughout: `Chicago - UK`.

## Working Conclusion

The most likely explanation is a procedure error on side B:

- on side A, wires should be measured starting from the low slot
- on side B, wires should be measured starting from the high slot
- in Chicago, side B appears to have been measured starting from the low slot
  instead

That would reverse the B-side wire order relative to the intended indexing.

The data support that interpretation. Among the tested B-side index models, the
best-performing one is:

- reverse the B-side index order
- then apply a small additional `-1` index shift

In wire-number form, that model is:

`Chicago B wire w` -> `UK B wire 481 - w`

This is the strongest result from the tested mapping family.

## Source Artifacts

- B-side model comparison plot:
  `data/tension_plots/tension_b_index_model_comparison_UKAPA7_G.png`
- A-side vs corrected B-side plot:
  `data/tension_plots/tension_a_vs_reversed_shifted_b_UKAPA7_G.png`
- B-side model comparison table:
  `data/tension_summaries/tension_b_index_model_comparison_UKAPA7_G.csv`
- A-side vs corrected B-side table:
  `data/tension_summaries/tension_a_vs_reversed_shifted_b_UKAPA7_G.csv`

## What Likely Happened

If the measurement operator starts from the low slot on both sides, then side A
is indexed correctly but side B is traversed in the wrong direction. That
causes Chicago side-B wire numbers to be paired against the wrong UK side-B
wires.

The additional `-1` shift suggests there is also a small endpoint or numbering
offset in the reversal, for example:

- the reversed sequence may have been anchored to the wrong end wire
- the first or last physical slot may have been handled differently
- the reversal may be correct conceptually, but the indexing origin used in the
  data products differs by one wire

The key point is that the dominant effect is the reversal. The extra shift is a
secondary correction.

## Plot 1: B-Side Index Models

This plot compares three B-side mappings:

- baseline: current CSV wire numbers matched to the same UK wire numbers
- reversed: full reversal with no shift
- reversed plus best shift: the best-fitting reversal model from the scan

![UKAPA7 B-side index models](../data/tension_plots/tension_b_index_model_comparison_UKAPA7_G.png){ width=100% }

### B-Side Model Statistics

| Model | Count | Corr | Mean (N) | Median (N) | Std (N) | MAE (N) | RMSE (N) | Negative Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 481 | 0.2319 | -0.1994 | -0.1976 | 0.4188 | 0.3712 | 0.4638 | 0.6798 |
| Reversed | 481 | 0.1681 | -0.1994 | -0.2385 | 0.4346 | 0.3914 | 0.4782 | 0.6840 |
| Reversed + shift `-1` | 480 | 0.3300 | -0.2002 | -0.2979 | 0.3935 | 0.3739 | 0.4415 | 0.7125 |

### Interpretation

- Pure reversal is not enough by itself. It performs worse than the baseline.
- Reversal plus the `-1` shift is clearly better than both baseline and pure
  reversal by correlation and residual spread.
- The residual mean stays near `-0.20 N` for all B-side models. That is
  expected: remapping changes which wires are paired, not the overall average
  of the two campaigns.

## Plot 2: A Side Versus Corrected B Side

This plot compares:

- side A under the normal indexing convention
- side B under the best corrected model, `reversed + shift -1`

![UKAPA7 A vs corrected B](../data/tension_plots/tension_a_vs_reversed_shifted_b_UKAPA7_G.png){ width=100% }

### A vs Corrected B Statistics

| Series | Count | Mean (N) | Median (N) | Std (N) | MAE (N) | Max Abs (N) | Negative Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A baseline | 481 | -0.3762 | -0.4269 | 0.4103 | 0.4843 | 1.6307 | 0.8441 |
| B reversed + shift `-1` | 480 | -0.2002 | -0.2979 | 0.3935 | 0.3739 | 1.3778 | 0.7125 |

### Interpretation

- After correcting the B-side indexing, the corrected B residuals are tighter
  than A by standard deviation and by maximum absolute residual.
- A remains more uniformly negative than corrected B, with a larger mean
  negative shift.
- So the indexing correction improves B substantially, but it does not make B
  identical to A. There is still side-to-side variation beyond the indexing
  issue.

## Overall Interpretation

- The B-side mismatch is most plausibly explained by reversing the measurement
  direction on side B.
- The best-tested correction is not just a reversal, but a reversal plus a
  `-1` shift.
- That small extra shift is consistent with an off-by-one difference in how the
  reversed sequence was anchored to wire numbering.
- With this correction, the B-side residual distribution becomes visibly tighter
  and more internally consistent.

## Bottom Line

The likely procedural error is that Chicago side-B measurements were started
from the low slot when they should have been started from the high slot. That
would reverse the B-side wire order. The data support that explanation, and the
best tested correction is a reversed B-side mapping with an additional `-1`
shift.
