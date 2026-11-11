# DistilBERT selective automation

**Best result: 45.1% acceptance and 92.8% accuracy among accepted cases on 1,200 reserved cases.**

- [Full trial report](pass13/README.md)
- [Best run](../../jev/selective-automation/best%20run/README.md)
- [Frozen policy](selected_policy.json)
- `run_trial.py`: calibration and evaluation runner (completed outputs protected)
- `verify.py`: independent reconstruction of all 1,200 held-out routing decisions

DistilBERT's validation-selected checkpoint supplies outcome predictions; JEV's intensity-of-care specialist overrides when its chosen outcome probability is at least 0.875. Otherwise accept DistilBERT at probability at least 0.90. Defer the rest. Ground truth remains recorded dataset decisions. DistilBERT predictions are cached historical outputs, not new inference; the final trial made 1,200 live JEV calls. This is the final one of ten additional authorized trials, stored separately as requested.
