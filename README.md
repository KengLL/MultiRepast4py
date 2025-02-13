# MultiRepast4py

MultiRepast4py is a Python package that enables multilayer simulations on Repast4py. It reconstructs given network files to incorporate multilayer simulation features. For more details, refer to the tutorials or demos in this repository.

---

## Installation

### Step 1: Install Repast4py

Follow the official guide to install Repast4py: [Repast4py Installation Guide](https://repast.github.io/repast4py.site/index.html)

### Step 2: Install MultiRepast4py

You can install MultiRepast4py using one of the following methods:

```bash
pip install git+https://github.com/KengLL/MultiRepast4py.git
```

or

```bash
pip install multirepast4py
```

---

## Usage

### Importing the Required Packages

```python
import networkx as nx
from multirepast4py.network import parse_and_write_network_files
from repast4py.network import write_network
```

### Create Networks (Currently Supporting NetworkX)

```python
# Parameters for the Erdős–Rényi graph
N = 10000  # Number of nodes
p = 0.005  # Probability of edge creation
rank = 1   # MPI rank

# Create two layers of networks
G1 = nx.erdos_renyi_graph(N, p)
G2 = nx.erdos_renyi_graph(N, p)

# Write network layers to files
write_network(G1, 'network', 'layer1.txt', rank)
write_network(G2, 'network', 'layer2.txt', rank)

# Parse and combine network layers for multilayer simulation
parse_and_write_network_files(['layer1.txt', 'layer2.txt'])
```

### Modify Agent Attributes

Add a `shadow_data` attribute to agents to store multilayer connections.

### Accessing Multilayer Edges in Agent Step Functions

Agents' outgoing edges can be accessed as follows:

```python
# Access the weight of an edge on a specific layer
weight = agent.shadow_data[layer].get(neighbor, None)

# Access all neighbors of an agent on a specific layer
neighbors = agent.shadow_data[layer].keys()
```

### Example: Rumor Network Model

This example extends the Rumor Network Model from the [Repast4py User Guide](https://repast.github.io/repast4py.site/guide/user_guide.html#_tutorial_2_the_rumor_network_model) to a multilayer network setup.

```python
from mpi4py import MPI
from repast4py import context as ctx, schedule, core, logging, parameters
from repast4py.network import read_network
from multirepast4py.network import parse_and_write_network_files
import numpy as np

class RumorAgent(core.Agent):
    def __init__(self, nid, agent_type, rank, received_rumor=False, shadow=None):
        super().__init__(nid, agent_type, rank)
        self.received_rumor = received_rumor
        self.shadow = shadow or {}

    def update(self, received_rumor, shadow_data):
        if not self.received_rumor and received_rumor:
            self.received_rumor = True
        self.shadow = shadow_data

class RumorModel:
    def __init__(self, comm, params):
        self.context = ctx.SharedContext(comm)
        read_network(params['network_file'], self.context, RumorAgent)
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_stop(params['stop.at'])
        self.rumor_prob = params['rumor_probability']

    def step(self):
        for agent in self.context.agents():
            for neighbor in agent.shadow_data[0].keys():  # Example for layer 0
                if np.random.rand() < self.rumor_prob:
                    neighbor.received_rumor = True

    def run(self):
        self.runner.execute()

if __name__ == "__main__":
    params = parameters.init_params('parameters.json')
    comm = MPI.COMM_WORLD
    model = RumorModel(comm, params)
    model.run()
```

## Execution
To execute the model using MPI (Parallel Computing for Performance), enter the following command in the terminal:
```bash
mpirun -n <rank> python <model_file> <params_file>
```
For users who do not require MPI, the code can be modified to support simple Python execution or Jupyter Notebook by replacing the execution flow with the following snippet:
```bash
if __name__ == "__main__":
    params = {
        'layer_schedules': [
            {'start': 1, 'interval': 1},
            {'start': 1, 'interval': 1}
        ],
        'network_file': 'networks/layer1.txt_multi',
        'initial_rumor_count': 1,
        'stop.at': 100,
        'rumor_probability': 0.01,
        'counts_file': 'output/rumor_counts.csv'
    }
    run(params)
```
---

## Key Features

- **Multilayer Network Simulation**: Extend Repast4py to handle multilayer networks.
- **Flexible Integration**: Easily incorporate NetworkX graphs into Repast4py.
- **Efficient Agent Management**: Access multilayer edges and neighbors during simulations.

---

## Resources

- **Repast4py Documentation**: [Official Docs](https://repast.github.io/repast4py.site/index.html)
- **Tutorials and Examples**: Check the `Demo` folder in this repository for detailed use cases.

---

## Contributing

Contributions are welcome! Please submit issues or pull requests for bug fixes, new features, or documentation improvements.
