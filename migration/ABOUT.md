# ABOUT THIS BRANCH
This demo we:
0. In `migration.toml`, set the local project directories (`crn_paper` and `starsim`) and SHA commits corresponding to the `starsim` versions.
1. Generate information about dependencies `crn_paper` has on starsim using `get_methods_list/test1.py` (model: `gpt-4o-mini`)
2. Aggregate the diffs and then generate full text solutions using `get_suggestions/test1.py`. The original scripts in `crn_paper/crn_paper` and `crn_paper/scripts` are overwritten so you can look at the diffs using e.g., github. (model: `gpt-4o`)

Quick notes:
- W/o additional edits, only `run_PPH.py` finishes albeit `run_comparison.py` is missing a critical import statement in the `starsim==1.0` version.
- I haven't looked at how many additional changes are necessary to get the other scripts to work.



