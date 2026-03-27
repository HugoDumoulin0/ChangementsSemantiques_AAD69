# imports

# dataclass permet de définir des petits objets structurés
# très pratiques pour représenter les énoncés et les relations φ.
from dataclasses import dataclass

# networkx sert à construire des graphes (nœuds + arêtes)
import networkx as nx

# matplotlib sert à afficher le graphe final
import matplotlib.pyplot as plt


# 1. data structures

@dataclass
class Enonce:
    """
    Représente un énoncé élémentaire (E1.1, E2.1, etc.).
    Chaque énoncé a :
      - un identifiant (index)
      - un texte brut (text)
    Ici, on ne garde que ce qui est nécessaire pour le graphe φ.
    """
    index: str
    text: str


@dataclass
class Relation:
    """
    Représente une relation φ entre deux énoncés.
    Chaque relation contient :
      - source : énoncé d'origine
      - target : énoncé d'arrivée
      - label  : le type de φ (φ3, φ8, φ4…)
    Ce modèle correspond exactement à la structure Θ du manuel.
    """
    source: str
    target: str
    label: str  # φ only


# 2. φ relations

# Que les énoncés qui participent
# aux relations φ. On représente donc uniquement :
#   E1.1 → E2.1 → E3.1
# qui sont les trois énoncés principaux du graphe Θ.
enonces = [
    Enonce("E1.1", "ils arrivèrent"),
    Enonce("E2.1", "le Roi et la Reine étaient assis"),
    Enonce("E3.1", "une grande foule entourait"),
]


# 3. φ relations theta 

# Les relations φ définissent l’enchaînement discursif :
#   φ3 = simultanéité ("Quand…")
#   φ8 = succession
#   φ4 = arrêt (point final)
#la structure Θ du manuel.
phi_relations = [
    Relation("E1.1", "E2.1", "φ3"),  # simultanéité
    Relation("E2.1", "E3.1", "φ8"),  # succession
    Relation("E3.1", "E3.1", "φ4"),  # arrêt (boucle)
]


# 4. Build graph

# On crée un graphe orienté (DiGraph) :
#   - les nœuds = énoncés
#   - les arêtes = relations φ
G = nx.DiGraph()

#ajout des nœuds (E1.1, E2.1, E3.1)
for e in enonces:
    # Chaque nœud porte aussi son texte comme attribut
    G.add_node(e.index, text=e.text)

#ajout des arêtes φ
for r in phi_relations:
    # Chaque arête porte une étiquette φ (φ3, φ8, φ4)
    G.add_edge(r.source, r.target, label=r.label)


# 5. Visualize graph theta

#on fixe manuellement la position des nœuds pour obtenir
# un graphe clair et linéaire
#   E1.1 → E2.1 → E3.1
pos = {
    "E1.1": (0, 0),
    "E2.1": (1, 0),
    "E3.1": (2, 0),
}

plt.figure(figsize=(10, 4))

#dessin des nœuds
nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="lightgreen")

#dessin des labels des nœuds (E1.1, E2.1…)
nx.draw_networkx_labels(G, pos, font_size=9)

#dessin des arêtes φ
nx.draw_networkx_edges(
    G, pos,
    arrowstyle="->",
    arrowsize=20,
    connectionstyle="arc3,rad=0.2"
)

#dessin des étiquettes φ (φ3, φ8, φ4)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

plt.title("Graphe Θ")
plt.axis("off")
plt.show()

