from load_graph import load_graph

G = load_graph()

for node, data in G.nodes(data=True):

    if data.get("node_type") == "transaction":

        print("Transaction node example:")
        print(data)
        break