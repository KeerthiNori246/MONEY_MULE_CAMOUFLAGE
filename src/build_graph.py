import pandas as pd
import networkx as nx
from tqdm import tqdm
from graph_schema import NODE_TYPES
from datetime import datetime

def build_transaction_graph(csv_path):
    df = pd.read_csv(csv_path)
    G = nx.Graph()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        customer = NODE_TYPES["customer"] + str(row["Customer_ID"])
        merchant = NODE_TYPES["merchant"] + str(row["Merchant_ID"])
        device = NODE_TYPES["device"] + str(row["Device_Type"])
        location = NODE_TYPES["location"] + str(row["City"])
        transaction = NODE_TYPES["transaction"] + str(row["Transaction_ID"])

        # Add nodes
        G.add_node(customer, node_type="customer")
        G.add_node(merchant, node_type="merchant")
        G.add_node(device, node_type="device")
        G.add_node(location, node_type="location")

        # Normalize timestamp BEFORE adding transaction node
        date_str = row["Transaction_Date"]
        time_str = row["Transaction_Time"]
        
        # Convert DD-MM-YYYY to YYYY-MM-DD
        try:
            parsed_date = datetime.strptime(date_str, "%d-%m-%Y")
            normalized_date = parsed_date.strftime("%Y-%m-%d")
        except:
            normalized_date = date_str  # Fallback if already correct

        # Transaction node with normalized attributes
        G.add_node(
            transaction,
            node_type="transaction",
            fraud=row["Is_Fraud"],
            amount=row["Transaction_Amount"],
            date=normalized_date,  # Now in YYYY-MM-DD format
            time=time_str
        )

        # Add edges
        G.add_edge(customer, transaction, edge_type="customer_transaction")
        G.add_edge(transaction, merchant, edge_type="transaction_merchant")
        G.add_edge(transaction, device, edge_type="transaction_device")
        G.add_edge(transaction, location, edge_type="transaction_location")

    return G
