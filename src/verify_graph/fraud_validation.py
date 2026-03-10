from load_graph import load_graph

def main():

    G = load_graph()

    fraud_count = 0
    legit_count = 0

    for node,data in G.nodes(data=True):

        if data["node_type"] == "transaction":

            if data.get("fraud") == 1:
                fraud_count += 1
            else:
                legit_count += 1

    print("Fraud transactions:", fraud_count)
    print("Legit transactions:", legit_count)

if __name__ == "__main__":
    main()