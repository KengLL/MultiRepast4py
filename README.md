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

---
## Package Structure

The new network package uses dataclasses for better structure and type safety:

### Core Classes

-   **`NetworkData`**: Main container for all network data
    
    -   `agents`: Dictionary of Agent objects by node ID
        
    -   `num_layers`: Total number of network layers
        
    -   `is_directed`: Whether the network is directed
        
-   **`Agent`**: Represents an agent in the multilayer network
    
    -   `node_id`: Unique identifier for the agent
        
    -   `agent_type`: Type of the agent
        
    -   `rank`: MPI rank
        
    -   `layers`: List of LayerData objects containing edge information
        
-   **`LayerData`**: Edge data for a single layer
    
    -   `layer_id`: Identifier for the layer
        
    -   `edges`: List of Edge objects
        
-   **`Edge`**: Represents a directed edge
    
    -   `target_id`: Target agent ID
        
    -   `weight`: Edge weight (default: 1.0)
        

### Key Methods

-   **`process_multilayer_network()`**: Main function to process multiple network files
    
-   **`NetworkFileProcessor`**: Handles file parsing and multilayer file generation
    
-   **`CompressionHandler`**: Compresses/decompresses agent data for efficient storage

  
## Usage

### Basic Multilayer Network Processing

```python
from multirepast4py.network import process_multilayer_network, NetworkData, Agent

# Process multiple network layers
file_paths = ['layer1.txt', 'layer2.txt', 'layer3.txt']
network_data = process_multilayer_network(file_paths)

# Access processed network data
print(f"Network has {len(network_data.agents)} agents across {network_data.num_layers} layers")
print(f"Network is directed: {network_data.is_directed}")

# Access individual agents
agent = network_data.get_agent(node_id=1)
if agent:
    print(f"Agent {agent.node_id} (type={agent.agent_type}, rank={agent.rank})")
    for layer in agent.layers:
        print(f"  Layer {layer.layer_id}: {len(layer.edges)} outgoing edges")
        for edge in layer.edges:
            print(f"    -> Agent {edge.target_id} (weight={edge.weight})")
```

### Create Networks with NetworkX

```python
import networkx as nx
from multirepast4py.network import process_multilayer_network
from repast4py.network import write_network

# Parameters for the Erdős–Rényi graph
N = 10000  # Number of nodes
p = 0.005  # Probability of edge creation
rank = 1   # MPI rank

# Create multiple network layers
G1 = nx.erdos_renyi_graph(N, p)
G2 = nx.erdos_renyi_graph(N, p)
G3 = nx.erdos_renyi_graph(N, p)

# Write network layers to files
write_network(G1, 'network', 'layer1.txt', rank)
write_network(G2, 'network', 'layer2.txt', rank)
write_network(G3, 'network', 'layer3.txt', rank)

# Process and combine network layers for multilayer simulation
network_data = process_multilayer_network(['layer1.txt', 'layer2.txt', 'layer3.txt'])
```

### Accessing Multilayer Connections in Agents

```python
def agent_step(agent):
    """Example agent step function accessing multilayer connections"""
    
    # Access edges in specific layers
    for layer in agent.layers:
        print(f"Processing layer {layer.layer_id}")
        
        for edge in layer.edges:
            target_id = edge.target_id
            weight = edge.weight
            
            # Your agent logic here
            if weight > 0.5:  # Example condition
                process_connection(agent, target_id, layer.layer_id, weight)
    
    # Get specific layer by ID
    layer_0 = agent.get_layer(0)
    if layer_0:
        print(f"Layer 0 has {len(layer_0.edges)} connections")
        
        # Access edges as dictionary for compatibility
        edges_dict = layer_0.get_edges_dict()
        for target_id, weight in edges_dict.items():
            process_important_connection(agent, target_id, weight)

def process_connection(source_agent, target_id, layer_id, weight):
    """Process a single connection"""
    # Your connection processing logic
    pass

def process_important_connection(source_agent, target_id, weight):
    """Process important connections"""
    # Your logic for important connections
    pass
```

### Example: Multilayer Rumor Spread Model

This example extends the Rumor Network Model from the [Repast4py User Guide](https://repast.github.io/repast4py.site/guide/user_guide.html#_tutorial_2_the_rumor_network_model) to a multilayer network setup.

```python
from mpi4py import MPI
from repast4py import context as ctx, schedule, core, logging, parameters
from repast4py.network import read_network
from multirepast4py.network import process_multilayer_network, Agent
import numpy as np

class MultilayerRumorAgent(core.Agent):
    def __init__(self, nid, agent_type, rank, received_rumor=False):
        super().__init__(nid, agent_type, rank)
        self.received_rumor = received_rumor
        # Multilayer data will be stored in the agent's layers attribute
        self.layers = []

    def update_network_data(self, network_data):
        """Update agent with multilayer network data"""
        full_agent = network_data.get_agent(self.id)
        if full_agent:
            self.layers = full_agent.layers

class MultilayerRumorModel:
    def __init__(self, comm, params):
        self.context = ctx.SharedContext(comm)
        
        # Process multilayer network first
        layer_files = ['layer1.txt', 'layer2.txt', 'layer3.txt']
        self.network_data = process_multilayer_network(layer_files)
        
        # Read the base network (multilayer data is stored in agent attributes)
        read_network('layer1.txt_multi', self.context, MultilayerRumorAgent)
        
        # Update agents with full multilayer data
        for agent in self.context.agents():
            agent.update_network_data(self.network_data)
            
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_stop(params['stop.at'])
        self.rumor_prob = params['rumor_probability']
        
        # Schedule the step function
        self.runner.schedule_repeating_event(1, 1, self.step)

    def step(self):
        """Model step function - rumor spread across multiple layers"""
        for agent in self.context.agents():
            if agent.received_rumor:
                # Spread rumor through all layers
                for layer in agent.layers:
                    for edge in layer.edges:
                        if np.random.rand() < self.rumor_prob * edge.weight:
                            target_agent = self.context.get_agent(edge.target_id)
                            if target_agent and not target_agent.received_rumor:
                                target_agent.received_rumor = True
                                print(f"Rumor spread from {agent.id} to {target_agent.id} on layer {layer.layer_id}")

    def run(self):
        self.runner.execute()

if __name__ == "__main__":
    # First, ensure you have network files, then:
    params = parameters.init_params('model_params.json')
    comm = MPI.COMM_WORLD
    model = MultilayerRumorModel(comm, params)
    model.run()
```

### Working with Adjacency Lists
```python
from multirepast4py.network import convert_to_adjacency_list

# Convert to adjacency list format for analysis
adjacency_list = convert_to_adjacency_list(network_data)

# Analyze each layer
for layer_id, edges in adjacency_list.items():
    print(f"Layer {layer_id}:")
    print(f"  Total edges: {len(edges)}")
    
    # Calculate average weight
    total_weight = sum(edge['weight'] for edge in edges)
    avg_weight = total_weight / len(edges) if edges else 0
    print(f"  Average weight: {avg_weight:.3f}")
    
    # Example: Find strong connections
    strong_edges = [e for e in edges if e['weight'] > 0.7]
    print(f"  Strong connections (weight > 0.7): {len(strong_edges)}")
```

## Execution
To execute the model using MPI (Parallel Computing for Performance), enter the following command in the terminal:
```bash
mpirun -n <rank> python <model_file> <params_file>
```
---

## Key Features

-   **Modern Architecture**: Uses Python dataclasses for type safety and cleaner code
    
-   **Efficient Storage**: Compresses multilayer data into agent attributes
    
-   **Flexible Access**: Multiple ways to access edge data (objects, dictionaries, adjacency lists)
    
-   **Seamless Integration**: Works with existing Repast4py networks and agents
    
-   **Layer Management**: Easy access to specific layers and their connections
---

## Resources

- **Repast4py Documentation**: [Official Docs](https://repast.github.io/repast4py.site/index.html)
- **Tutorials and Examples**: Check the `Demo` folder in this repository for detailed use cases.

---

## Contributing

Contributions are welcome! Please submit issues or pull requests for bug fixes, new features, or documentation improvements.
