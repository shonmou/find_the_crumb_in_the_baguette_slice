# boundary-refinement Specification

## Purpose
TBD - created by archiving change update-grabcut-default. Update Purpose after archive.
## Requirements
### Requirement: GrabCut Boundary Refinement By Default

The script SHALL run GrabCut boundary refinement on every image unless explicitly suppressed. GrabCut SHALL run as the final pipeline step after watershed edge refinement, using the watershed mask as its initialisation. The result SHALL replace the watershed mask as the primary contour source.

#### Scenario: GrabCut active by default

- **WHEN** the user runs `python identify_bread_area.py bread.jpg` without any refinement flags
- **THEN** the progress line "Refining boundary with GrabCut …" appears in stdout and the output mask reflects GrabCut refinement

#### Scenario: GrabCut suppressed with --no-grabcut

- **WHEN** the user runs `python identify_bread_area.py bread.jpg --no-grabcut`
- **THEN** GrabCut is skipped, the pipeline ends after watershed refinement, and processing completes faster

#### Scenario: GrabCut fallback on degenerate mask

- **WHEN** GrabCut raises `cv2.error` due to a degenerate initialisation mask
- **THEN** the script falls back to the watershed-derived mask and continues without error

### Requirement: Deprecated --grabcut Flag Removed

The `--grabcut` opt-in flag SHALL be removed. Any script or command that previously passed `--grabcut` MUST drop that flag; passing it SHALL produce an `argparse` error.

#### Scenario: Old --grabcut flag rejected

- **WHEN** the user passes `--grabcut` on the command line
- **THEN** `argparse` prints an "unrecognised arguments" error and exits with a non-zero code

