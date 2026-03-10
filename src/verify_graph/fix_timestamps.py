from load_graph import load_graph
from normalize_timestamps import normalize_timestamps
import pickle

GRAPH_PATH = "../../graphs/transaction_graph.gpickle"

def main():
    print("Loading graph...")
    G = load_graph()
    
    print("Normalizing timestamps...")
    G = normalize_timestamps(G)
    
    print("Saving normalized graph...")
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    
    print("Done! Graph overwritten with normalized dates.")

if __name__ == "__main__":
    main()
