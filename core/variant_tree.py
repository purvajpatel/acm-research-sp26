"""
Variant tree data structure for recursive problem decomposition
Each node has exactly one parent, maintaining clear difficulty progression
"""
import uuid
from typing import List, Optional, Dict

class VariantNode:
    """
    Represents a single variant in the problem decomposition tree
    """
    def __init__(self, problem: str, depth: int = 0, parent: Optional['VariantNode'] = None):
        self.id = str(uuid.uuid4())
        self.problem = problem
        self.depth = depth
        self.parent = parent
        self.children: List['VariantNode'] = []
        self.solution: Optional[str] = None
        self.is_verified: bool = False
        self.verification_result: Optional[bool] = None
    
    def add_child(self, child_node: 'VariantNode'):
        """Add a child variant node"""
        child_node.parent = self
        self.children.append(child_node)
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)"""
        return len(self.children) == 0
    
    def get_all_descendants(self) -> List['VariantNode']:
        """Get all descendant nodes"""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants
    
    def __repr__(self):
        return f"VariantNode(id={self.id[:8]}, depth={self.depth}, problem={self.problem[:50]}...)"

class VariantTree:
    """
    Tree structure for organizing problem variants
    Maintains parent-child relationships for difficulty progression
    """
    def __init__(self):
        self.nodes: List[VariantNode] = []
        self.root: Optional[VariantNode] = None
        self.max_depth: int = 0
    
    def add_node(self, node: VariantNode):
        """Add a node to the tree"""
        self.nodes.append(node)
        if node.depth == 0:
            self.root = node
        self.max_depth = max(self.max_depth, node.depth)
    
    def get_node(self, node_id: str) -> Optional[VariantNode]:
        """Get node by ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_nodes_by_depth(self, depth: int) -> List[VariantNode]:
        """Get all nodes at a specific depth"""
        return [node for node in self.nodes if node.depth == depth]
    
    def get_leaves(self) -> List[VariantNode]:
        """Get all leaf nodes (simplest variants)"""
        return [node for node in self.nodes if node.is_leaf()]
    
    def get_levels(self) -> List[List[VariantNode]]:
        """Get all nodes organized by depth level"""
        levels = []
        for depth in range(self.max_depth + 1):
            level_nodes = self.get_nodes_by_depth(depth)
            if level_nodes:
                levels.append(level_nodes)
        return levels
    
    def get_path_to_root(self, node: VariantNode) -> List[VariantNode]:
        """Get path from node to root"""
        path = [node]
        current = node.parent
        while current is not None:
            path.append(current)
            current = current.parent
        return path
    
    def __len__(self):
        return len(self.nodes)
    
    def __repr__(self):
        return f"VariantTree(root={self.root is not None}, nodes={len(self.nodes)}, max_depth={self.max_depth})"
