Calibration tool: check_calib
=============================

Purpose
-------

Check_calib is a deprecated tool aiming at validating both the geometric
calibration and everything else like flat-field correction, distortion
correction, at a sub-pixel level. Pleas use the `validate`, `validate2` and
`chiplot` commands in pyFAI-calib, during the refinement process to obtain the same output with more options.

.. command-output:: check_calib --help
    :nostderr:
