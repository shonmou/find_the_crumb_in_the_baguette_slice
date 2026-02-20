## ADDED Requirements

### Requirement: Ellipse Protrusion Clipping

After GrabCut refinement the pipeline SHALL apply an ellipse-based protrusion filter to remove angular foreign objects (such as cardboard wedges) that are physically connected to the bread and cannot be separated by colour or connected-component analysis. The filter SHALL fit a minimum-enclosing ellipse to the final contour and intersect the filled mask with the ellipse, clipping any protrusions that extend beyond the smooth elliptical boundary.

#### Scenario: Cardboard wedge removed

- **WHEN** the image contains a cardboard object touching the bread that was included in the GrabCut mask
- **THEN** the final `_bread_exact.png` and mask do not contain the cardboard protrusion

#### Scenario: Clean image unchanged

- **WHEN** the image contains bread with no foreign objects
- **THEN** the detected area changes by less than 2% compared to the pre-filter result, and all output files are produced successfully

#### Scenario: Fallback on non-elliptical or degenerate mask

- **WHEN** the fitted ellipse would reduce the detected area by more than 25%, or the contour has fewer than 5 points
- **THEN** the ellipse clip is skipped and the unclipped mask is used as-is
