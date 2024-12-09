import os
import json
import inspect
import tomllib
from pathlib import Path
import sciris as sc
import starsim_ai as sa
import crn_paper
import starsim as ss
import traceback
import subprocess

os.chdir(os.path.dirname(__file__))

MODEL = 'gpt-4o-mini'

prompt_str = """
NOTHING YET
"""

# gets the git diff of a file between two commits
def get_diff(file, t: dict, padding: int = 10):
    with sa.utils.TemporaryDirectoryChange(t['info']['starsim']):
        # cmd = """git diff --patience {} {} -- {} | grep -A {} -B {} 'function_name'""".format(t['info']['commit1'], t['info']['commit2'], file, padding, padding)
        cmd = [s for s in """git diff --patience {} {} -- {}""".format(t['info']['commit1'], t['info']['commit2'], file).split()]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=t['info']['starsim'])
    return result.stdout

def get_all_diffs(file, mehtods_list):
    diffs = {}
    # just keep track of the files that have changes
    for method in methods_list:
        if method.startswith("ss."):
            method_str = method.split("ss.", 1)[1]
        else:
            method_str = method
        try:
            attr = getattr(ss, method_str)
            attr_file = inspect.getfile(attr)
            stdout = get_diff(attr_file, t)
            if attr_file not in diffs:
                diffs[attr_file] = stdout
        except AttributeError:
            print(f"Attribute {method_str} not found")
        except Exception as e:
            print(f"traceback: {traceback.format_exc()}")
    return diffs

if __name__ == "__main__":

    migration_dir = crn_paper.paths.src / 'migration'

    # open the file
    with open(migration_dir / 'migration.toml','rb') as f:
        t = tomllib.load(f)
    code_dir = Path(t['info']['code']).resolve()

    # open the list of methods
    with open(migration_dir / f'results/identified_ss_{MODEL}.json','r') as f:
        methods_dict = json.load(f)

    # looping over the project files and methods found
    for file, methods_list in methods_dict.items():
        print(f"File: {file}")
        print(f"Methods: {methods_list}")

        # get the diffs relevant to the code file
        diffs = get_all_diffs(file, methods_list)
        print(f"Number of diffs: {len(diffs)}")

        if len(diffs) > 0:
            # construct the prompt.
            pass
   
