## ADDED Requirements

### Requirement: Mat Marking Exclusion

After the inverted mat detection (`find_bread_within_mat`) produces the initial bread candidate mask, the pipeline SHALL apply `suppress_background` to remove near-white ruler tick marks and grid numbers that pass through the inverted detection. The suppression SHALL run before morphological refinement so that markers do not survive the closing step.

#### Scenario: Ruler markings excluded from candidate mask

- **WHEN** the image contains near-white ruler tick marks or grid numbers on the cutting mat surface
- **THEN** those pixels do not appear in the bread candidate mask passed to `refine_mask`

#### Scenario: No regression when markings are absent

- **WHEN** the image has no visible ruler markings in the mat area
- **THEN** the resulting bread mask is equivalent to the pre-suppression result and the detected bread area is within 5% of the baseline measurement

### Requirement: Wood-Grain Desktop Exclusion

The pipeline SHALL remove warm low-saturation desktop pixels (H 8–40, S < 62, V > 120) that fall inside the mat convex hull — which can occur when the hull spans gaps between mat fragments over the wood surface.

#### Scenario: Wood grain inside hull boundary excluded

- **WHEN** the convex hull of teal contours encloses some wood-grain desktop pixels
- **THEN** those pixels are removed from the bread candidate mask and do not contribute to the detected contour

#### Scenario: Bread crust not affected by wood exclusion

- **WHEN** the image contains bread crust in the warm-brown hue range (S > 65)
- **THEN** crust pixels are retained because the wood-grain exclusion requires S < 62
