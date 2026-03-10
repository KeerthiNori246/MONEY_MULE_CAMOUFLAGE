from load_graph import load_graph

VALID_TYPES = {
    "customer",
    "merchant",
    "device",
    "location",
    "transaction"
}

def main():

    G = load_graph()

    errors = 0

    for node,data in G.nodes(data=True):

        if "node_type" not in data:
            print("Missing node_type:", node)
            errors += 1
            continue

        if data["node_type"] not in VALID_TYPES:
            print("Invalid node_type:", node, data)
            errors += 1

    print("Schema errors:", errors)

if __name__ == "__main__":
    main()