import os
import re
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
import tiktoken

os.chdir(os.path.dirname(__file__))

MODEL = 'gpt-4o'

prompt_template = '''
Here is the diff information for an update to the starsim (ss) package:
------
{}
------
Please update the code below to maintain compatibility with the starsim (ss) code:
------
{}
------
Return ONLY the updated code:
'''

def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    # encoding = tiktoken.get_encoding(encoding_name)
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

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
            pass
            # print(f"Attribute {method_str} not found")
        except Exception as e:
            pass
            # print(f"traceback: {traceback.format_exc()}")
    return diffs

if __name__ == "__main__":

    migration_dir = crn_paper.paths.src / 'migration'

    # open the file
    with open(migration_dir / 'migration.toml','rb') as f:
        t = tomllib.load(f)
    code_dir = Path(t['info']['code']).resolve()

    # open the list of methods
    with open(migration_dir / f'results/identified_ss_gpt-4o-mini.json','r') as f:
        methods_dict = json.load(f)

    # create the query object
    query = sa.SimpleQuery(model=MODEL)

    # looping over the project files and methods found
    for file, methods_list in methods_dict.items():
        print(f"File: {file}")
        print(f"Methods: {methods_list}")

        # get the diffs relevant to the code file
        diffs = get_all_diffs(file, methods_list)
        print(f"Number of diffs: {len(diffs)}")

        # load the file
        code_file = Path(t['info']['code']).resolve() / file
        with open(code_file, 'r') as f:
            code_string = f.read()

        if len(diffs) > 0:
            # construct the prompt
            prompt = prompt_template.format("\n".join(diffs.values()), code_string)
            print(f"Number of tokens: {num_tokens_from_string(prompt, MODEL)}")
            response = query(prompt)
            pattern = "```python(.*?)```"
            matches = re.findall(pattern, response.content, re.DOTALL)
            with open(migration_dir / 'results' / f'{code_file.stem}_{MODEL}.py', 'w') as f:
                f.write(matches[0])
            with open(code_file, 'w') as f:
                f.write(matches[0])


   
