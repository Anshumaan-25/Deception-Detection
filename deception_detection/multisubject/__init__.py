"""Multi-subject tooling — built 2026-07-10, ahead of the N>1 corpus arriving 2026-07-11.

Two tools (full workflow: Documentation/MULTISUBJECT_REPLICATION_PLAN.md):
  intake_validator.py       validate a new subject's package (videos + ELAN .eaf)
                            BEFORE any GPU time is spent on a malformed session
  replication_scorecard.py  after per-subject cascades: does SubjectA's per-channel
                            signal replicate across subjects, against criteria that
                            were PRE-REGISTERED before the new data was seen?

Doctrine unchanged: ELAN labels are used for SCORING ONLY — never calibration,
never training. Pure pandas/numpy/stdlib; runs on laptop and desktop alike.
"""
