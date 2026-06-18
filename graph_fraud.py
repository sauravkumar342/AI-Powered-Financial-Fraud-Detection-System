import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

from config import RAW_DATA_PATH, OUTPUT_DIR


def graph_fraud_detection():
    print("\n🔗 Building Transaction Graph...")

    df = pd.read_csv(RAW_DATA_PATH)

    # 🚀 speed optimization (important)
    df = df.sample(n=50000, random_state=42)

    # Use only important columns
    df = df[["nameOrig", "nameDest", "amount", "isFraud"]]

    # Create graph
    G = nx.DiGraph()

    for _, row in df.iterrows():
        sender = row["nameOrig"]
        receiver = row["nameDest"]
        amount = row["amount"]
        fraud = row["isFraud"]

        G.add_edge(sender, receiver, weight=amount, fraud=fraud)

    print("✅ Graph created")

    # 🔥 Detect suspicious nodes (high degree)
    degrees = dict(G.degree())
    suspicious_nodes = sorted(degrees, key=degrees.get, reverse=True)[:10]

    print("\n🚨 Top Suspicious Nodes (High Activity):")
    for node in suspicious_nodes:
        print(node, "→ connections:", degrees[node])

    # 🔥 Visualization
    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(G, k=0.15)

    # color edges
    edge_colors = [
        'red' if G[u][v]['fraud'] == 1 else 'gray'
        for u, v in G.edges()
    ]

    nx.draw_networkx_nodes(G, pos, node_size=10)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.5)

    plt.title("Transaction Graph (Red = Fraud)")
    plt.axis('off')

    save_path = os.path.join(OUTPUT_DIR, "transaction_graph.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📊 Graph saved at: {save_path}")