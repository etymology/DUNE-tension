# UKAPA7 Layer U Partial B-Side Offset Report

This report repeats the UKAPA7 comparison workflow for the U layer using:

- UK factory measurements in `UKAPA7U.json`
- Chicago factory measurements in
  `data/tension_summaries/tension_summary_UKAPA7_U.csv`

Residual definition throughout: `Chicago - UK`.

## Data Availability

- The UK JSON contains both side A and side B measurements.
- The Chicago summary CSV now contains all currently available U-layer data on
  both sides.
- The available Chicago side-B wires span wire numbers `241` through `967`.
- The current CSV contains `641` non-null side-B measurements and `385` aligned
  side-A comparison rows.

This report still focuses on the available side-B subset, since the U-layer
indexing question being tested here is a B-side offset model.

## Working Assumption

For this U-layer analysis, the working assumption is:

- there is no side-B reversal
- there may be a constant wire-number offset

So the tested model family is a simple shift:

`Chicago B wire w` -> `UK B wire w + k`

with no reversal term.

## Executive Summary

- Side A data are now available in the Chicago summary and are included below as
  a direct raw-data comparison.
- For the currently available A-side rows, the mean residual is about
  `-0.544 N`.
- Under the current indexing, the partial B-side comparison is already strongly
  negative on average, with mean residual about `-0.871 N`.
- Scanning constant shifts from `-80` to `+80` shows the best correlation at
  shift `-7`.
- That shift improves correlation from `0.509` to `0.586` and reduces the
  residual width from `0.548 N` to `0.492 N`.
- The improvement is real but not transformative. The Chicago and UK traces
  remain substantially offset even after the best tested small shift.

## Source Artifacts

- Raw A comparison:
  `data/tension_plots/tension_raw_A_UKAPA7_U.png`
- Raw B comparison, current indexing:
  `data/tension_plots/tension_raw_B_baseline_UKAPA7_U.png`
- Raw B comparison, best offset:
  `data/tension_plots/tension_raw_B_shifted_UKAPA7_U.png`
- Residual comparison, baseline vs best offset:
  `data/tension_plots/tension_residual_B_offset_comparison_UKAPA7_U.png`
- Offset scan:
  `data/tension_plots/tension_shift_scan_B_UKAPA7_U.png`
- Aligned comparison table:
  `data/tension_summaries/tension_B_offset_comparison_UKAPA7_U.csv`

## Raw A-Side Data

![UKAPA7 U raw A](../data/tension_plots/tension_raw_A_UKAPA7_U.png){ width=100% }

### A-Side Summary

| Comparison | Count | Chicago Mean (N) | UK Mean (N) | Corr | Mean Diff (N) | Std Diff (N) | MAE (N) | Negative Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Side A | 385 | 5.419 | 5.962 | 0.104 | -0.544 | 0.524 | 0.610 | 0.855 |

### Interpretation

- Side A is now available as a direct UK-versus-Chicago comparison.
- The mean residual is negative, consistent with the same overall direction of
  change seen elsewhere.
- The A-side correlation is still weak in the currently available subset, so the
  report keeps its main focus on the B-side offset question rather than trying
  to build an indexing model for side A.

## Raw B-Side Data, Current Indexing

![UKAPA7 U raw B baseline](../data/tension_plots/tension_raw_B_baseline_UKAPA7_U.png){ width=100% }

### Current-Indexing Summary

| Comparison | Count | Chicago Mean (N) | UK Mean (N) | Corr | Mean Diff (N) | Std Diff (N) | MAE (N) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Side B baseline | 641 | 5.331 | 6.202 | 0.509 | -0.871 | 0.548 | 0.895 |

## Raw B-Side Data, Best Constant Offset

![UKAPA7 U raw B shifted](../data/tension_plots/tension_raw_B_shifted_UKAPA7_U.png){ width=100% }

### Best-Offset Summary

| Comparison | Count | Chicago Mean (N) | UK Mean (N) | Corr | Mean Diff (N) | Std Diff (N) | MAE (N) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Side B shift `-7` | 641 | 5.331 | 6.200 | 0.586 | -0.869 | 0.492 | 0.874 |

### Interpretation

- A simple negative shift of `-7` wires produces the best correlation in the
  tested range.
- The improvement is visible in both the line plot and the histogram overlap.
- Even after the shift, the Chicago values remain systematically below the UK
  values by about `0.95 N` on average.

## Residual Comparison

This figure compares the residuals for:

- baseline indexing
- the best tested constant offset, shift `-7`

![UKAPA7 U residual comparison](../data/tension_plots/tension_residual_B_offset_comparison_UKAPA7_U.png){ width=100% }

### Residual Statistics

| Model | Count | Corr | Mean (N) | Median (N) | Std (N) | MAE (N) | RMSE (N) | Negative Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 641 | 0.509 | -0.871 | -0.810 | 0.548 | 0.895 | 1.029 | 0.972 |
| Shift `-7` | 641 | 0.586 | -0.869 | -0.756 | 0.492 | 0.874 | 0.999 | 0.989 |

### Interpretation

- The best constant shift tightens the residual distribution and increases the
  correlation.
- The residuals remain strongly negative even after the shift, which means the
  dominant effect is still a large average Chicago-versus-UK difference.
- Since only part of side B is present in the CSV, this should be interpreted
  as a partial-window comparison, not a statement about the entire layer.

## Offset Scan

![UKAPA7 U shift scan](../data/tension_plots/tension_shift_scan_B_UKAPA7_U.png){ width=100% }

### Interpretation

- The correlation curve peaks near shift `-7`.
- The residual standard deviation also reaches a local minimum around the same
  region.
- Larger negative shifts can lower MAE somewhat, but they do not give the best
  correlation and are less compelling as a simple indexing-offset explanation.

## Conclusion

For the available U-layer data, there is no need to invoke a reversal model.
The Chicago summary now has partial data on both sides, and the newly available
A-side rows are included as a direct raw comparison. Within the side-B window,
the best simple index-offset explanation remains a shift of `-7` wires.

That shift improves the match, but only moderately. The comparison still shows
an overall large negative offset between Chicago and UK measurements, and the
fact that the U-layer coverage is still partial limits how far the
interpretation can be pushed.
