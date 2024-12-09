import os
import json
import tomllib
from pathlib import Path
import sciris as sc
import starsim_ai as sa
import crn_paper

os.chdir(os.path.dirname(__file__))

MODEL = 'gpt-4o-mini'

migration_dir = crn_paper.paths.src / 'migration'

# open the file
with open(migration_dir / 'migration.toml','rb') as f:
    t = tomllib.load(f)
code_dir = Path(t['info']['code']).resolve()

# get all python files in our project
python_files = sa.get_python_files(code_dir, gitignore=True)
print(f"Found {len(python_files)} Python files")

prompt_string = """
In the following Python code, find all dependencies on the {} package, 
including inherited methods and attributes from parent classes. Do not list starsim or ss itself.
List only the dependencies from starsim.
-------------
Here is an example:

Code:
```python
import starsim as ss

# Define the parameters
pars = dict(
    n_agents = 5_000,     # Number of agents to simulate
    networks = dict(      # Networks define how agents interact w/ each other
        type = 'random',  # Here, we use a 'random' network
        n_contacts = 10   # Each person has 10 contacts with other people
    ),
    diseases = dict(      # *Diseases* add detail on what diseases to model
        type = 'sir',     # Here, we're creating an SIR disease
        init_prev = 0.01, # Proportion of the population initially infected
        beta = 0.05,      # Probability of transmission between contacts
    )
)

# Make the sim, run and plot
sim = ss.Sim(pars)
sim.run()
sim.plot() # Plot all the sim results
sim.diseases.sir.plot() # Plot the standard SIR curves
```
Answer:
['ss.Sim', 'ss.Sim.run', 'ss.Sim.plot', 'ss.diseases', 'ss.diseases.sir', 'ss.diseases.sir.plot']
-------------

Now, it's  your turn. Please be as accurate as possible.

Code:

```python
{}
```

Answer:
"""

query = sa.CSVQuery(model=MODEL, temperature=0.0)
res = {}
sc.tic()
# figure out all the references to starsim
for file in python_files:
    print("Checking file:", file)
    with open(code_dir / file, 'r') as f:
        text = f.read()
    ans = query(prompt_string.format("starsim (ss)", text))
    ans = [a.replace('`', '') for a in ans] # remove backticks
    res[file] = ans
sc.toc()

with open(migration_dir / f'results/identified_ss_{MODEL}.json','w') as f:
    json.dump(res,f, indent=2)

