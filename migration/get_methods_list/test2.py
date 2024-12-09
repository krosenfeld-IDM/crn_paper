"""
https://python.langchain.com/docs/how_to/structured_output/#the-with_structured_output-method
"""
import os
import tomllib
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field
import starsim_ai as sa
import sciris as sc

os.chdir(os.path.dirname(__file__))

# Pydantic
class Methods(BaseModel):
    """Methods to investigate"""

    methods_list: str = Field(description="List of methods to investigate")


query = sa.BaseQuery(model='gpt-4o-mini')
query.llm = query.llm.with_structured_output(Methods)


# open the file
with open('migration.toml','rb') as f:
    t = tomllib.load(f)
code_dir = Path(t['info']['code']).resolve()

# get all python files in our project
python_files = sa.get_python_files(code_dir, gitignore=True)
print(f"Found {len(python_files)} Python files")

prompt_string = """
In the following Python code, find all dependencies on the {} package, 
including inherited methods and attributes from parent classes. 
List only the dependencies.
-------------
Code:

```python
{}
```

Answer:
"""

res = {}
sc.tic()
# figure out all the references to starsim
for file in python_files:
    print("Checking file:", file)
    with open(code_dir / file, 'r') as f:
        text = f.read()
    ans = query(prompt_string.format("starsim (ss)", text)) 
    res[file] = ans
sc.toc()
