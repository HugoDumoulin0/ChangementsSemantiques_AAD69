#We import dataclasses to define simple structured objects (Eᵢ and φ‑relations)
from dataclasses import dataclass

#NetworkX is the library to build and manipulate graphs
import networkx as nx

import matplotlib.pyplot as plt

# 1. DATA STRUCTURES/ linguistic model
#these classes represent the "surface discursive" structure as the manual describes it.


@dataclass
class Enonce:
    """
    Represents one elementary énoncé (Eᵢ).
    Each énoncé has:
      - an index (1, 2, 3…)
      - the raw text
      - the 8-slot linguistic encoding used in the manual:
            F, D1, N1, V, ADV, PP, D2, N23
    """
    index: int
    text: str
    F: str = None
    D1: str = None
    N1: str = None
    V: str = None
    ADV: str = None
    PP: str = None
    D2: str = None
    N23: str = None


@dataclass
class PhiRelation:
    """
    Represents a φ-relation between two énoncés.
    source → target with a label φ₁…φ₈.
    This corresponds to the 'Ensemble B₃' table in the manual.
    """
    source: int
    target: int
    phi: str


# 2.create ÉNONCÉS manually encoding 3 clauses
# These are simplified versions of the Alice text.
# We fill only the essential slots (F, N1, V, etc.to demonstrate how the system works.


E1 = Enonce(
    1,
    "Quand ils arrivèrent",
    F="F4",
    N1="ils",
    V="arriver"
)

E2 = Enonce(
    2,
    "le Roi et la Reine étaient assis sur des trônes",
    F="F4",
    D1="le",
    N1="Roi+Reine",
    V="être assis",
    PP="sur des trônes",
    D2="des",
    N23="trônes"
)

E3 = Enonce(
    3,
    "Une grande foule les entourait",
    F="F4",
    D1="une",
    N1="foule",
    V="entourer"
)


# 3.hard-code φ RELATIONS reproducing the manual
#these φ-relations correspond to:
#   E1 → E2 : φ₃ (simultaneity: "Quand…")
#   E2 → E3 : φ₈ (next in sequence)
#   E3 → E3 : φ₄ (stop: ".")
#how the manual encodes the discursive surface.


relations = [
    PhiRelation(1, 2, "phi3"),   # simultaneity
    PhiRelation(2, 3, "phi8"),   # next in sequence
    PhiRelation(3, 3, "phi4")    # stop
]


# 4. build the θ-graph 
# We now create a directed graph (DiGraph).
# Nodes = énoncés (Eᵢ)
# Edges = φ-relations (φ₁…φ₈)


G = nx.DiGraph()

#each énoncé as a node in the graph
for E in [E1, E2, E3]:
    G.add_node(E.index, text=E.text)

#each φ-relation as a labeled edge
for r in relations:
    G.add_edge(r.source, r.target, phi=r.phi)



# 5. inspect the graph
# This shows the nodes and edges like the manual’s tables.


print("Nodes:")
print(G.nodes(data=True))

print("\nEdges:")
print(G.edges(data=True))

# 6. VISUALIZE THE GRAPH

pos = nx.spring_layout(G)  # automatic layout

#draw nodes
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue")

#draw edges
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20)

#draw node labels (E1, E2, E3)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

#draw φ labels on edges
edge_labels = nx.get_edge_attributes(G, 'phi')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

plt.title("φ‑Graph (Theta Graph) of Elementary Énoncés")
plt.axis("off")
plt.show()
