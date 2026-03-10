from load_graph import load_graph
from datetime import datetime

def main():

    G = load_graph()

    errors = 0

    for node,data in G.nodes(data=True):

        if data["node_type"] == "transaction":

            try:

                datetime.strptime(
                    data["date"],
                    "%Y-%m-%d"
                )

            except:

                errors += 1

    print("Timestamp errors:", errors)

if __name__ == "__main__":
    main()