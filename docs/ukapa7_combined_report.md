# UKAPA7 Combined Comparison Report

This report compares two full wire tension measurements for UKAPA7 layer G:

- UK factory measurements in `apa_uk7g.json` on December 10, 2023
- Chicago factory measurements in
  `data/tension_summaries/tension_summary_UKAPA7_G.csv` on March 11, 2026

Residual definition throughout: `Chicago - UK`.

## Executive Summary

- Side A has the expected sign: Chicago tensions are lower than UK tensions on
  average, with mean residual `-0.376 N`.
- The Chicago B side (on this and all previous APAs) has been measured with reversed order indices. For comparison we reverse the indices of the B side measurement.

## Physical Expectation

The expected physical change from the UK campaign to the Chicago campaign is a
roughly constant negative offset, since the later Chicago tensions should tend
to be lower.

Both measurement campaigns are noisy. If the single-measurement uncertainty is
about `0.25 N` to `0.5 N`, then the residual scatter expected from two
independent noisy measurements is roughly `0.35 N` to `0.71 N`.

The observed residual widths are around `0.41 N` to `0.43 N`, which is
compatible with that noise scale. The main issue is therefore how side-B wires
are paired between the two campaigns.

## Source Artifacts

- Raw side-A comparison:
  `data/tension_plots/tension_raw_A_UKAPA7_G.png`
- Raw side-B comparison, reversed indexing:
  `data/tension_plots/tension_raw_B_reversed_UKAPA7_G.png`
- A-side versus reversed-B residuals:
  `data/tension_plots/tension_a_vs_reversed_b_UKAPA7_G.png`

## Raw Data Plots

These plots show the original UK and Chicago tensions directly. Each plot uses
the same format:

- top panel: per-wire scatter with rolling trendlines
- bottom panel: overlaid histograms
- statistics shown directly on the plot

### Side A Raw Tensions

![UKAPA7 raw side A](../data/tension_plots/tension_raw_A_UKAPA7_G.png){ width=100% }

### Side B Raw Tensions, Reversed Indexing

![UKAPA7 raw side B reversed](../data/tension_plots/tension_raw_B_reversed_UKAPA7_G.png){ width=100% }

### Raw Data Summary

| Comparison | Count | Chicago Mean (N) | UK Mean (N) | Corr | Mean Diff (N) | Std Diff (N) | MAE (N) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Side A | 481 | 5.761 | 6.137 | 0.227 | -0.376 | 0.410 | 0.484 |
| Side B reversed | 481 | 5.849 | 6.048 | 0.168 | -0.199 | 0.435 | 0.391 |

### Interpretation

- Side A shows a stable negative offset between the UK and Chicago campaigns.
- The reversed-B plot is included because it matches the procedural hypothesis:
  if Chicago side B was measured from the low slot instead of the high slot,
  then the B-side ordering would be reversed.

## A Side Versus Reversed B Side

This figure compares:

- side A under the normal indexing convention
- side B under the reversed-index hypothesis implied by the measurement
  procedure

![UKAPA7 A vs reversed B](../data/tension_plots/tension_a_vs_reversed_b_UKAPA7_G.png){ width=100% }

### A Versus Reversed B Statistics

| Series | Count | Mean (N) | Median (N) | Std (N) | MAE (N) | Max Abs (N) | Negative Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A baseline | 481 | -0.3762 | -0.4269 | 0.4103 | 0.4843 | 1.6307 | 0.8441 |
| B reversed | 481 | -0.1994 | -0.2385 | 0.4346 | 0.3914 | 1.5501 | 0.6840 |

### Interpretation

- Under the reversed-B interpretation, side B remains less negative than side A
  and has a somewhat broader residual distribution.
- That means the reversal alone does not make side B look identical to side A.
- Even so, reversal is the right procedural hypothesis to visualize, because it
  reflects what would happen if the Chicago side-B scan started from the wrong
  end.


## Conclusion

The combined evidence supports the following interpretation:

- the overall UK-to-Chicago change has the expected negative sign
- side A is broadly consistent with that picture
- the most relevant procedural explanation is that side B was measured in the
  reverse direction, starting from the low slot instead of the high slot

In practical terms, the operational fix is straightforward: measure side A from
the low slot and side B from the high slot, and verify that the side-B wire
order in the output follows that convention.
