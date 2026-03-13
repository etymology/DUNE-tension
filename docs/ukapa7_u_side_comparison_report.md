# UKAPA7 Layer U Chicago vs UK Comparison Report

This report compares the current UKAPA7 U-layer tension measurements from:

- UK factory JSON data in `UKAPA7U.json`
- Chicago summary CSV data in `data/tension_summaries/tension_summary_UKAPA7_U.csv`

Residual definition throughout: `Chicago - UK`.

## Data Availability

- The UK JSON contains `1141` non-null tensions on both side A and side B.
- The current Chicago summary CSV contains `385` side-A values and `641` side-B
  values.
- All non-null Chicago values align to non-null UK values at the same wire
  number.
- The aligned wire ranges are `500` through `940` on side A and `241` through
  `967` on side B.

This is therefore a direct same-index comparison over the currently available
Chicago subset on each side, not a full-layer comparison over all U wires.

## Executive Summary

- Side A currently has `385` aligned wires. Its residuals are broadly negative,
  with mean `-0.544 N`, standard deviation `0.524 N`, and weak but positive
  correlation (`0.104`) between the Chicago and UK measurements.
- Side B currently has `641` aligned wires. It is also systematically negative,
  with mean residual `-0.871 N`, standard deviation `0.548 N`, and noticeably
  stronger correlation (`0.509`) than side A.
- Both sides therefore show Chicago tensions below the UK tensions on average,
  but side B is more coherent as a trace while side A is noisier and less
  correlated in the currently populated window.

## Source Artifacts

- Comparison table:
  `data/tension_summaries/tension_comparison_UKAPA7_U.csv`
- Raw A-side plot:
  `data/tension_plots/tension_raw_A_UKAPA7_U.png`
- Raw B-side plot:
  `data/tension_plots/tension_raw_B_UKAPA7_U.png`
- Combined residual plot:
  `data/tension_plots/tension_residual_UKAPA7_U.png`

## Summary Statistics

| Side | Aligned Wires | Wire Range | Chicago Mean (N) | UK Mean (N) | Corr | Mean Diff (N) | Median Diff (N) | Std Diff (N) | MAE (N) | RMSE (N) | Negative Fraction |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 385 | `500-940` | 5.419 | 5.962 | 0.104 | -0.544 | -0.505 | 0.524 | 0.610 | 0.755 | 0.855 |
| B | 641 | `241-967` | 5.331 | 6.202 | 0.509 | -0.871 | -0.810 | 0.548 | 0.895 | 1.029 | 0.972 |

## Raw Side-A Comparison

![UKAPA7 U raw A](../data/tension_plots/tension_raw_A_UKAPA7_U.png){ width=100% }

### Interpretation

- The Chicago A-side mean is about `0.544 N` below the UK A-side mean over the
  populated Chicago window.
- The trace-level correlation is still weak, but it is no longer negative in
  the current A-side subset.
- The residuals are mostly negative and somewhat tighter than in the previous
  partial A-side comparison, but side A should still be treated as a noisy
  comparison rather than a nearly constant shift.

## Raw Side-B Comparison

![UKAPA7 U raw B](../data/tension_plots/tension_raw_B_UKAPA7_U.png){ width=100% }

### Interpretation

- Side B shows a larger average Chicago-versus-UK deficit, about `0.871 N`.
- Unlike side A, side B preserves a visible trace shape, reflected in the
  stronger correlation (`0.509`).
- The negative fraction is `0.972`, so the B-side residuals are almost entirely
  below zero in the currently available Chicago window.

## Residual Comparison

![UKAPA7 U residuals](../data/tension_plots/tension_residual_UKAPA7_U.png){ width=100% }

### Interpretation

- Both sides are biased negative, consistent with Chicago measurements being
  lower than the corresponding UK measurements in the available U-layer data.
- Side A has the smaller mean shift but the broader and less structured
  residual pattern.
- Side B has the larger mean shift, but its residuals are more clustered and
  more consistent with the underlying UK trace.

## Conclusion

The current UKAPA7 U-layer comparison shows the same sign on both sides:
Chicago tensions are lower than UK tensions on average. Side A and side B do
not behave equally, though. In the available A-side window, the comparison is
noisy and only weakly correlated wire-by-wire. In the larger B-side window, the
Chicago data track the UK structure more clearly, but with a larger overall
negative offset.

This report is intentionally limited to the currently populated U-layer rows in
the Chicago summary CSV. As more U-layer wires are added, the same plots and
statistics should be regenerated before drawing stronger conclusions about the
full layer.
