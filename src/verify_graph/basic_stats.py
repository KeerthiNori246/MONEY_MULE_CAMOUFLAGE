from load_graph import load_graph
from collections import Counter

def main():

    G = load_graph()

    print("Total nodes:", G.number_of_nodes())
    print("Total edges:", G.number_of_edges())

    node_types = [d["node_type"] for _, d in G.nodes(data=True)]

    counts = Counter(node_types)

    print("\nNode Type Counts")

    for k,v in counts.items():
        print(k, v)

if __name__ == "__main__":
    main()