import pickle
import os

# FIXED PATH from temporal/ perspective
GRAPH_PATH = os.path.abspath("../../graphs/transaction_graph.gpickle")

def load_graph():
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")
    
    print(f"Loading from: {GRAPH_PATH}")
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    return G
