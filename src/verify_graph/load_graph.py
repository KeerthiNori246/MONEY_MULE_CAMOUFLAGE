import pickle

GRAPH_PATH = "../../graphs/transaction_graph.gpickle"

def load_graph():

    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    return G