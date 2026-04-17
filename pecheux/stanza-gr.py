import stanza  # Import Stanza to handle the linguistic parsing (tokenization, POS, dependencies)
import networkx as nx  # Import NetworkX to build the mathematical Graph structure for Θ
import matplotlib.pyplot as plt  # Import Matplotlib to actually draw and visualize the final graph
import grewpy
from grewpy import Corpus, Request  # Import Grew tools to find φ-relations using patterns


# ---------------------------------------------------------
# 1. STANZA: Building the δ (Delta) Structure
# ---------------------------------------------------------
# This section creates the syntactic foundation (the tree).
stanza.download("fr")  # Download the French linguistic model (needed once to understand French)
nlp = stanza.Pipeline("fr")  # Initialize the French pipeline (Parser, Lemmatizer, etc.)

text = "Quand ils arrivèrent, le Roi et la Reine étaient assis sur des trônes, et une grande foule les entourait."  # Our raw input data
doc = nlp(text)  # Run the text through Stanza to create the syntactic dependency tree (δ)

# We convert Stanza's output into a CoNLL-U string (The "Export" step)
# This string represents the syntactic tree (δ) in a 10-column format Grew can read.
conllu_data = ""  # Start with an empty string to hold our exported data
for sent in doc.sentences:  # Loop through every sentence found in the text
    # Map every word's ID, Text, Lemma, POS, Head, and Relation into a tab-separated CoNLL-U line
    conllu_data += "".join([str(word.id) + "\t" + word.text + "\t" + word.lemma + "\t" + word.upos + "\t_\t_\t" + str(word.head) + "\t" + word.deprel + "\t_\t_\n" for word in sent.words])
    conllu_data += "\n"  # ← Blank line required between sentences in CoNLL-U

# ---------------------------------------------------------
# 2. GREWPY: Detecting φ (Phi) Relations
# ---------------------------------------------------------
# Here, Grew scans the δ-tree to find the "Discursive Articulations."

grewpy.init()  # Start the Grew Engine (the "motor" that performs the pattern matching)
# Write CoNLL-U to a temp file — Corpus reads from files, not raw strings
# Corpus() does not accept a raw Python string directly.
# It only knows how to read from a file path on disk.
# So we must write our CoNLL-U string to a real file first,
# then hand that file's path to Corpus(), then delete the file.
import tempfile, os

# Create a temporary file on disk that:
#   - opens in write mode ("w") so we can write text to it
#   - has the .conllu extension so grewpy recognizes the format
#   - has delete=False so the file isn't destroyed before Corpus() reads it
with tempfile.NamedTemporaryFile(mode="w", suffix=".conllu", delete=False) as f:
    f.write(conllu_data) # Write our CoNLL-U string into the file
    tmp_path = f.name # Save the file's path (e.g. /tmp/tmpXYZ.conllu)

# Now that the file exists on disk, Corpus() can read it by path
corpus = Corpus(tmp_path)  
# The file has been loaded into memory — we no longer need it on disk.
# os.unlink() permanently deletes it so we don't leave junk temp files behind.
os.unlink(tmp_path)        


# PATTERN A: Simultaneity
# Quand (mark) depends ON arrivèrent (advcl), so the arrow goes V -> M (head to dependent)
pattern_simultaneity = Request('pattern { V [deprel="advcl"]; M [form="Quand", deprel="mark"]; V -> M }')

# PATTERN B: Succession  
# entourait (conj) has head=assis, and et (cc) has head=entourait
# So the arrow goes V -> C (the conj verb governs its own cc)
pattern_succession = Request('pattern { V [deprel="conj"]; C [form="et", deprel="cc"]; V -> C }')

# ---------------------------------------------------------
# 3. BUILDING THE Θ (Theta) GRAPH
# ---------------------------------------------------------
# Now we take the relations found by Grew and turn them into a Narrative Map.
theta_graph = nx.DiGraph()  # Create a new Directed Graph (arrows move in one direction)

# Check the δ-tree for Simultaneity patterns
matches_sim = corpus.search(pattern_simultaneity)  # Ask Grew to find matches for Pattern A
if matches_sim:  # If Grew finds "Quand + Verb"
    # Create an edge in Θ from the subordinate event to the main event
    # In our text: "arrivèrent" (sub) -> "étaient" (main)
    theta_graph.add_edge("arrivèrent", "assis", label="simultaneity (φ3)")

# Check the δ-tree for Succession patterns
matches_succ = corpus.search(pattern_succession)  # Ask Grew to find matches for Pattern B
if matches_succ:  # If Grew finds "Verb + et + Verb"
    # Create an edge in Θ showing one event follows the other
    # In our text: "étaient" (main) -> "entourait" (coordinated)
    theta_graph.add_edge("assis", "entourait", label="succession (φ8)")

#debug
print("CoNLL-U output:\n", conllu_data)
print("Simultaneity matches:", matches_sim)
print("Succession matches:", matches_succ)

# Add the final Arrêt (φ4)
# By convention, the final event of a segment loops to itself to show the narrative stops
theta_graph.add_edge("entourait", "entourait", label="arrêt (φ4)")

# ---------------------------------------------------------
# 4. VISUALIZATION
# ---------------------------------------------------------
# This section turns our math data into a picture.
pos = nx.spring_layout(theta_graph)  # Position the nodes so they don't overlap too much
plt.figure(figsize=(10,6))  # Define the size of the drawing window
# Draw the nodes (circles) and the word labels inside them
nx.draw(theta_graph, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=12, arrowsize=20)
# Grab the specific φ-labels (Simultaneity, etc.) we added earlier
edge_labels = nx.get_edge_attributes(theta_graph, 'label')
# Draw these labels directly onto the arrows in the graph
nx.draw_networkx_edge_labels(theta_graph, pos, edge_labels=edge_labels, font_color='red')

plt.title("The Θ Graph: Narrative Structure")  # Give the window a clear title
plt.show()  # Pop open the window to display the final result


# =========================================================
# RÉSUMÉ DU WORKFLOW
# =========================================================
# Étape          | Outil      | Type de Sortie        | Nom théorique
# ---------------------------------------------------------
# Entrée         | Texte brut | Phrase complexe       | Le Corpus
# Sortie 1       | Stanza     | Arbre de dépendances  | Structure δ (Delta)
# Sortie 2       | GrewPy     | Liste de relations    | Relations φ (Phi)
# Sortie Finale  | NetworkX   | Graphe visuel         | Structure Θ (Theta)
# =========================================================