from load_graph import load_graph

def main():

    G = load_graph()

    transaction_nodes = [
        n for n,d in G.nodes(data=True)
        if d["node_type"] == "transaction"
    ]

    bad = 0

    for t in transaction_nodes:

        neighbors = list(G.neighbors(t))

        types = [
            G.nodes[n]["node_type"]
            for n in neighbors
        ]

        if not {"customer","merchant","device","location"}.issubset(types):

            bad += 1

    print("Transactions with missing edges:", bad)

if __name__ == "__main__":
    main()