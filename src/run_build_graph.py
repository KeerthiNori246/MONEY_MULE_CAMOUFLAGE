import pickle
from build_graph import build_transaction_graph

CSV_PATH = "../data/bank_fraud.csv"
OUTPUT_PATH = "../graphs/transaction_graph.gpickle"


def main():

    print("Building transaction graph...")

    G = build_transaction_graph(CSV_PATH)

    print("Graph built.")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    print("Saving graph...")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(G, f)

    print("Graph saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()