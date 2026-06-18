import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

from config import RAW_DATA_PATH, OUTPUT_DIR


def advanced_graph_analysis():
    print("\n🧠 Advanced Graph Analysis Started...")

    df = pd.read_csv(RAW_DATA_PATH)

    # ⚡ speed control
    df = df.sample(n=30000, random_state=42)

    df = df[["nameOrig", "nameDest", "amount", "isFraud"]]

    G = nx.DiGraph()

    for _, row in df.iterrows():
        G.add_edge(
            row["nameOrig"],
            row["nameDest"],
            weight=row["amount"],
            fraud=row["isFraud"]
        )

    print("✅ Graph Built")

    # ============================================
    # 🔥 1. COMMUNITY DETECTION
    # ============================================
    print("\n🔍 Detecting Communities...")

    undirected_G = G.to_undirected()
    communities = nx.algorithms.community.greedy_modularity_communities(undirected_G)

    print(f"✅ Total Communities Found: {len(communities)}")

    # Save community info
    community_file = os.path.join(OUTPUT_DIR, "communities.txt")
    with open(community_file, "w") as f:
        for i, comm in enumerate(communities):
            f.write(f"Community {i}: {list(comm)[:10]}\n")

    print(f"📁 Communities saved: {community_file}")

    # ============================================
    # 🚨 2. FRAUD RING DETECTION
    # ============================================
    print("\n🚨 Detecting Fraud Rings...")

    fraud_edges = [(u, v) for u, v, d in G.edges(data=True) if d["fraud"] == 1]

    fraud_graph = nx.DiGraph()
    fraud_graph.add_edges_from(fraud_edges)

    cycles = list(nx.simple_cycles(fraud_graph))

    print(f"⚠️ Fraud Rings Found: {len(cycles)}")

    ring_file = os.path.join(OUTPUT_DIR, "fraud_rings.txt")
    with open(ring_file, "w") as f:
        for i, cycle in enumerate(cycles[:20]):
            f.write(f"Ring {i}: {cycle}\n")

    print(f"📁 Fraud rings saved: {ring_file}")

    # ============================================
    # 🤖 3. PSEUDO GNN FEATURES (Lightweight)
    # ============================================
    print("\n🤖 Generating Graph-Based Features...")

    degree_centrality = nx.degree_centrality(G)
    pagerank = nx.pagerank(G)

    features = []

    for node in G.nodes():
        features.append({
            "node": node,
            "degree": degree_centrality.get(node, 0),
            "pagerank": pagerank.get(node, 0)
        })

    feat_df = pd.DataFrame(features)

    feature_file = os.path.join(OUTPUT_DIR, "graph_features.csv")
    feat_df.to_csv(feature_file, index=False)

    print(f"📊 Graph features saved: {feature_file}")

    # ============================================
    # 📊 VISUALIZATION
    # ============================================
    print("\n📊 Saving Advanced Graph Visualization...")

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.2)

    edge_colors = [
        'red' if G[u][v]['fraud'] == 1 else 'gray'
        for u, v in G.edges()
    ]

    nx.draw(G, pos,
            node_size=10,
            edge_color=edge_colors,
            alpha=0.6,
            with_labels=False)

    save_path = os.path.join(OUTPUT_DIR, "advanced_graph.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📈 Graph image saved: {save_path}")