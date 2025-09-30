"""
Multilayer Network Processing Package for Repast4py Simulations

This module provides utilities for processing multilayer network files,
managing agent data, and handling edge information across multiple layers.
The refactored version uses dataclasses for better structure and type safety,
and employs a cleaner data format using lists instead of nested dictionaries.
"""

import json
import zlib
import base64
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import warnings


# ========================
# Data Structures
# ========================

@dataclass
class Edge:
    """Represents a directed edge in the network."""
    target_id: int
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {"target": self.target_id, "weight": self.weight}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Edge':
        """Create edge from dictionary representation."""
        return cls(target_id=data["target"], weight=data.get("weight", 1.0))


@dataclass
class LayerData:
    """
    Represents edge data for a single layer.
    Using a list-based approach instead of nested dictionaries.
    """
    layer_id: int
    edges: List[Edge] = field(default_factory=list)
    
    def add_edge(self, target_id: int, weight: float = 1.0) -> None:
        """Add an edge to this layer."""
        self.edges.append(Edge(target_id, weight))
    
    def get_edges_dict(self) -> Dict[int, float]:
        """Get edges as a dictionary for compatibility."""
        return {edge.target_id: edge.weight for edge in self.edges}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert layer data to dictionary representation."""
        return {
            "layer_id": self.layer_id,
            "edges": [edge.to_dict() for edge in self.edges]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerData':
        """Create layer data from dictionary representation."""
        return cls(
            layer_id=data["layer_id"],
            edges=[Edge.from_dict(e) for e in data.get("edges", [])]
        )


@dataclass
class Agent:
    """
    Represents an agent in the multilayer network.
    Uses a list of LayerData objects instead of nested dictionaries.
    """
    node_id: int
    agent_type: int
    rank: int
    layers: List[LayerData] = field(default_factory=list)
    
    def get_layer(self, layer_id: int) -> Optional[LayerData]:
        """Get layer data by ID, returns None if not found."""
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer
        return None
    
    def add_layer(self, layer_id: int) -> LayerData:
        """Add or get a layer by ID."""
        layer = self.get_layer(layer_id)
        if layer is None:
            layer = LayerData(layer_id=layer_id)
            self.layers.append(layer)
            # Keep layers sorted by ID for consistency
            self.layers.sort(key=lambda l: l.layer_id)
        return layer
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation."""
        return {
            "node_id": self.node_id,
            "agent_type": self.agent_type,
            "rank": self.rank,
            "layers": [layer.to_dict() for layer in self.layers]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Create agent from dictionary representation."""
        return cls(
            node_id=data["node_id"],
            agent_type=data["agent_type"],
            rank=data["rank"],
            layers=[LayerData.from_dict(l) for l in data.get("layers", [])]
        )


@dataclass
class NetworkData:
    """Container for all network data including agents and metadata."""
    agents: Dict[int, Agent] = field(default_factory=dict)
    num_layers: int = 0
    is_directed: bool = True
    
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the network."""
        self.agents[agent.node_id] = agent
    
    def get_agent(self, node_id: int) -> Optional[Agent]:
        """Get agent by node ID."""
        return self.agents.get(node_id)
    
    def add_edge(self, source_id: int, target_id: int, layer_id: int, 
                 weight: float = 1.0) -> None:
        """Add an edge between agents in a specific layer."""
        if source_id in self.agents:
            agent = self.agents[source_id]
            layer = agent.add_layer(layer_id)
            layer.add_edge(target_id, weight)
            
            # Add reverse edge if undirected
            if not self.is_directed and target_id in self.agents:
                target_agent = self.agents[target_id]
                target_layer = target_agent.add_layer(layer_id)
                target_layer.add_edge(source_id, weight)


# ========================
# Compression Utilities
# ========================

class CompressionHandler:
    """Handles compression and decompression of agent data."""
    
    @staticmethod
    def compress_agent_data(agent: Agent) -> str:
        """
        Compress agent layer data to base64-encoded string.
        
        Args:
            agent: Agent object containing layer data
            
        Returns:
            Base64-encoded compressed string
        """
        data = {"layers": [layer.to_dict() for layer in agent.layers]}
        json_str = json.dumps(data, separators=(',', ':'))
        compressed = zlib.compress(json_str.encode('utf-8'))
        return base64.b64encode(compressed).decode('utf-8')
    
    @staticmethod
    def decompress_agent_data(encoded_str: str) -> List[LayerData]:
        """
        Decompress base64-encoded string to layer data.
        
        Args:
            encoded_str: Base64-encoded compressed string
            
        Returns:
            List of LayerData objects
        """
        try:
            compressed = base64.b64decode(encoded_str)
            json_str = zlib.decompress(compressed).decode('utf-8')
            data = json.loads(json_str)
            return [LayerData.from_dict(l) for l in data.get("layers", [])]
        except Exception as e:
            warnings.warn(f"Failed to decompress agent data: {e}")
            return []


# ========================
# File Processing
# ========================

class NetworkFileProcessor:
    """Processes multilayer network files."""
    
    def __init__(self, file_paths: List[str]):
        """
        Initialize processor with file paths.
        
        Args:
            file_paths: List of paths to network files
        """
        self.file_paths = [Path(p) for p in file_paths]
        self.network_data = NetworkData(num_layers=len(file_paths))
    
    def parse_files(self) -> NetworkData:
        """
        Parse all network files and build network data structure.
        
        Returns:
            NetworkData object containing all agents and edges
        """
        # First pass: parse nodes from first file
        if self.file_paths:
            self._parse_nodes(self.file_paths[0])
        
        # Second pass: parse edges from all files
        for layer_id, file_path in enumerate(self.file_paths):
            self._parse_edges(file_path, layer_id)
        
        return self.network_data
    
    def _parse_nodes(self, file_path: Path) -> None:
        """Parse node definitions from the first file."""
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
        
        if not lines:
            return
        
        # Parse header for directed/undirected info
        header = lines[0].strip().split()
        self.network_data.is_directed = header[1] == '1' if len(header) > 1 else False
        
        # Parse nodes until EDGES marker
        for line in lines[1:]:
            if line.strip() == 'EDGES':
                break
            
            parts = line.strip().split(' ', 3)
            if len(parts) >= 3:
                node_id = int(parts[0])
                agent_type = int(parts[1])
                rank = int(parts[2])
                
                agent = Agent(node_id=node_id, agent_type=agent_type, rank=rank)
                
                # Parse existing compressed data if present
                if len(parts) > 3:
                    try:
                        attribs = json.loads(parts[3])
                        if 'data' in attribs:
                            agent.layers = CompressionHandler.decompress_agent_data(
                                attribs['data']
                            )
                    except (json.JSONDecodeError, KeyError):
                        pass
                
                self.network_data.add_agent(agent)
    
    def _parse_edges(self, file_path: Path, layer_id: int) -> None:
        """Parse edges from a network file."""
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
        
        found_edges = False
        for line in lines:
            if line.strip() == 'EDGES':
                found_edges = True
                continue
            
            if found_edges:
                parts = line.strip().split(' ', 2)
                if len(parts) >= 2:
                    source_id = int(parts[0])
                    target_id = int(parts[1])
                    
                    # Parse weight if present
                    weight = 1.0
                    if len(parts) > 2:
                        try:
                            attrs = json.loads(parts[2])
                            weight = attrs.get('weight', 1.0)
                        except (json.JSONDecodeError, KeyError):
                            pass
                    
                    self.network_data.add_edge(
                        source_id, target_id, layer_id, weight
                    )
    
    def write_multilayer_file(self, output_path: Optional[str] = None) -> Path:
        """
        Write network data to a new multilayer file.
        
        The output file contains only the node definitions with compressed multilayer
        edge data stored as agent attributes. The original EDGES section is removed
        since all edge information is now contained in the compressed data.
        
        Args:
            output_path: Optional output file path. If None, appends '_multi' to base file.
            
        Returns:
            Path to the written file
        """
        if output_path is None:
            base_path = self.file_paths[0]
            output_path = base_path.parent / f"{base_path.stem}_multi{base_path.suffix}"
        else:
            output_path = Path(output_path)
        
        # Copy base file as template
        shutil.copy(self.file_paths[0], output_path)
        
        with open(output_path, 'r') as f:
            lines = f.read().splitlines()
        
        updated_lines = []
        node_index = 0
        found_edges = False
        
        for line in lines:
            if line.strip() == 'EDGES':
                found_edges = True
                # Don't include the EDGES marker or any subsequent edge data
                # since all edge information is now stored in compressed agent attributes
                break
            
            if node_index == 0:  # Header line
                updated_lines.append(line)
            else:
                parts = line.strip().split(' ', 3)
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    agent = self.network_data.get_agent(node_id)
                    
                    if agent:
                        # Compress and encode agent data
                        encoded_data = CompressionHandler.compress_agent_data(agent)
                        attributes = json.dumps(
                            {"data": encoded_data}, 
                            separators=(',', ':')
                        )
                        new_line = f"{parts[0]} {parts[1]} {parts[2]} {attributes}"
                        updated_lines.append(new_line)
                    else:
                        updated_lines.append(line)
            node_index += 1
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(updated_lines))
        
        print(f"Updated network data written to: {output_path}")
        return output_path


# ========================
# High-Level API
# ========================

def process_multilayer_network(file_paths: List[str], 
                              output_path: Optional[str] = None) -> NetworkData:
    """
    Main function to process multilayer network files.
    
    Args:
        file_paths: List of paths to network files (one per layer)
        output_path: Optional output file path
        
    Returns:
        NetworkData object containing processed network
        
    Example:
        >>> files = ['layer0.net', 'layer1.net', 'layer2.net']
        >>> network = process_multilayer_network(files)
        >>> # Access agents and their connections
        >>> agent = network.get_agent(node_id=1)
        >>> if agent:
        >>>     for layer in agent.layers:
        >>>         print(f"Layer {layer.layer_id}: {len(layer.edges)} edges")
    """
    processor = NetworkFileProcessor(file_paths)
    network_data = processor.parse_files()
    processor.write_multilayer_file(output_path)
    return network_data


def convert_to_adjacency_list(network_data: NetworkData) -> Dict[int, List[Dict[str, Any]]]:
    """
    Convert network data to adjacency list format for easy traversal.
    
    Args:
        network_data: NetworkData object
        
    Returns:
        Dictionary mapping layer_id to list of edges
        
    Example:
        >>> adj_list = convert_to_adjacency_list(network)
        >>> # adj_list[0] contains all edges in layer 0
        >>> # Each edge is a dict with 'source', 'target', 'weight'
    """
    adjacency = defaultdict(list)
    
    for agent in network_data.agents.values():
        for layer in agent.layers:
            for edge in layer.edges:
                adjacency[layer.layer_id].append({
                    'source': agent.node_id,
                    'target': edge.target_id,
                    'weight': edge.weight
                })
    
    return dict(adjacency)


# ========================
# Example Usage
# ========================

if __name__ == "__main__":
    # Example usage
    file_paths = ["multirepast4py/networks/layer1.txt", "multirepast4py/networks/layer2.txt", "multirepast4py/networks/layer3.txt"]
    
    # Process the network files
    network = process_multilayer_network(file_paths)
    
    # Access agent data
    for node_id, agent in network.agents.items():
        print(f"Agent {node_id} (type={agent.agent_type}, rank={agent.rank})")
        for layer in agent.layers:
            print(f"  Layer {layer.layer_id}: {len(layer.edges)} outgoing edges")
            for edge in layer.edges[:3]:  # Show first 3 edges
                print(f"    -> Node {edge.target_id} (weight={edge.weight})")
    
    # Convert to adjacency list if needed
    adj_list = convert_to_adjacency_list(network)
    for layer_id, edges in adj_list.items():
        print(f"Layer {layer_id}: {len(edges)} total edges")
    
    print("\n" + "="*60)
    print("SANITY CHECK: Testing compression/decompression roundtrip")
    print("="*60)
    
    # Test compression and decompression for first few agents
    test_agents = list(network.agents.values())[:3]
    
    for agent in test_agents:
        print(f"\nTesting Agent {agent.node_id}:")
        
        # Original data
        original_layers = len(agent.layers)
        original_edges = sum(len(layer.edges) for layer in agent.layers)
        print(f"  Original: {original_layers} layers, {original_edges} total edges")
        
        # Compress the agent data
        compressed_data = CompressionHandler.compress_agent_data(agent)
        print(f"  Compressed size: {len(compressed_data)} characters")
        
        # Decompress the data
        decompressed_layers = CompressionHandler.decompress_agent_data(compressed_data)
        decompressed_layer_count = len(decompressed_layers)
        decompressed_edge_count = sum(len(layer.edges) for layer in decompressed_layers)
        
        print(f"  Decompressed: {decompressed_layer_count} layers, {decompressed_edge_count} total edges")
        
        # Verify data integrity
        if (original_layers == decompressed_layer_count and 
            original_edges == decompressed_edge_count):
            print("  ✅ PASS: Layer and edge counts match")
            
            # Detailed verification - check each layer
            all_match = True
            for orig_layer in agent.layers:
                # Find corresponding decompressed layer
                decomp_layer = None
                for dl in decompressed_layers:
                    if dl.layer_id == orig_layer.layer_id:
                        decomp_layer = dl
                        break
                
                if decomp_layer is None:
                    print(f"    ❌ FAIL: Layer {orig_layer.layer_id} not found in decompressed data")
                    all_match = False
                    continue
                
                # Check edge count
                if len(orig_layer.edges) != len(decomp_layer.edges):
                    print(f"    ❌ FAIL: Layer {orig_layer.layer_id} edge count mismatch")
                    all_match = False
                    continue
                
                # Check individual edges
                orig_edges = {(e.target_id, e.weight) for e in orig_layer.edges}
                decomp_edges = {(e.target_id, e.weight) for e in decomp_layer.edges}
                
                if orig_edges != decomp_edges:
                    print(f"    ❌ FAIL: Layer {orig_layer.layer_id} edge content mismatch")
                    all_match = False
                else:
                    print(f"    ✅ Layer {orig_layer.layer_id}: {len(orig_layer.edges)} edges match exactly")
            
            if all_match:
                print("  🎉 PERFECT: All layer data matches exactly!")
            
        else:
            print("  ❌ FAIL: Count mismatch detected")
    
    print(f"\n" + "="*60)
    print("Sanity check complete!")
    print("="*60)