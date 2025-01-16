import multinetX as mx  # Assuming 'mx' is a necessary module for multilayer graphs

def label(layers: list):
    """Labels nodes in each layer with 'layer' and 'number' attributes."""
    N = layers[0].number_of_nodes()
    for num in range(N):
        for layer in range(len(layers)):
            layers[layer].nodes[num]['layer'] = layer
            layers[layer].nodes[num]['number'] = num
    return layers

def create_multilayer_graph(label_layers):
    """Creates a multilayer graph from labeled layers."""
    mg = mx.MultilayerGraph()
    for layer in label_layers:
        mg.add_layer(layer)

    num_layers = len(label_layers)
    nodes_per_layer = label_layers[0].number_of_nodes()

    # Add edges directly using a generator expression to save memory
    for layer in range(num_layers - 1):
        start = layer * nodes_per_layer
        next_start = (layer + 1) * nodes_per_layer
        mg.add_edges_from((start + node, next_start + node) for node in range(nodes_per_layer))
    
    return mg

def categorize_agents(model_instance):
    """Categorizes agents into dynamically named layer sets based on their layer attribute and assigns them to the model instance."""
    if not hasattr(model_instance, 'context'):
        raise AttributeError("The model instance lacks a 'context' attribute.")
    
    # Dictionary to temporarily store each layer's agents
    layer_dict = {}

    # Categorize agents by their layer attribute
    for agent in model_instance.context.agents():
        layer = agent.layer
        # Initialize the set for
