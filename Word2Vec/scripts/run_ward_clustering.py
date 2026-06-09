import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA


DEFAULT_MATRICES_DIR = "ppmi_matrix/"
DEFAULT_OUTPUT_DIR = "ward_viz"
DEFAULT_MAX_K = 10
DEFAULT_N_CLUSTERS = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Genere les visualisations de clustering Ward.")
    parser.add_argument("--matrices-dir", default=DEFAULT_MATRICES_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Affiche les figures matplotlib a la fin.",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="N'affiche pas les figures, utile pour l'interface.",
    )
    return parser.parse_args()


def plot_dendrogram(model, labels=None, **kwargs):
    """
    Cette fonction est retrouvable sur sklearn.
    """
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, labels=labels, **kwargs)


def run_ward_clustering(
    matrices_dir=DEFAULT_MATRICES_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    max_k=DEFAULT_MAX_K,
    n_clusters=DEFAULT_N_CLUSTERS,
    show=True,
):
    os.makedirs(output_dir, exist_ok=True)

    global_matrix_path = os.path.join(matrices_dir, "cooc_global.npy")
    print(f"\nChargement de la matrice globale -> {global_matrix_path}")
    cooc_global = np.load(global_matrix_path)

    x_used = cooc_global
    vocab = np.arange(x_used.shape[0]).astype(str)

    inerties = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(x_used)
        inerties.append(kmeans.inertia_)

    elbow_path = os.path.join(output_dir, "elbow_method.jpeg")
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inerties, marker="o")
    plt.title("Methode du coude (Inertie)")
    plt.xlabel("Nombre de clusters (K)")
    plt.ylabel("Inertie (variance intra-cluster)")
    plt.xticks(list(k_range))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(elbow_path, format="jpeg", dpi=300, bbox_inches="tight")
    print(f"Courbe du coude sauvegardee -> {elbow_path}")
    if show:
        plt.show()
    else:
        plt.close()

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage="ward")
    model = model.fit(x_used)

    dendrogram_path = os.path.join(output_dir, "ward_dendrogram.jpeg")
    plt.figure(figsize=(24, 12))
    plt.title("Dendrogramme - Clustering Hierarchique (Ward)")
    plot_dendrogram(model, labels=vocab, leaf_rotation=90, leaf_font_size=6)
    plt.xlabel("Observations")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(dendrogram_path, format="jpeg", dpi=300, bbox_inches="tight")
    print(f"Dendrogramme sauvegarde -> {dendrogram_path}")
    if show:
        plt.show()
    else:
        plt.close()

    model_ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    cluster_labels = model_ward.fit_predict(x_used)

    pca_2d = PCA(n_components=2, random_state=42)
    x_2d = pca_2d.fit_transform(x_used)

    clusters_path = os.path.join(output_dir, "ward_clusters_pca.jpeg")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(x_2d[:, 0], x_2d[:, 1], c=cluster_labels, cmap="tab10", s=30, alpha=0.8)
    plt.title("Clusters Ward visualises avec PCA (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(clusters_path, format="jpeg", dpi=300, bbox_inches="tight")
    print(f"Projection PCA des clusters sauvegardee -> {clusters_path}")
    if show:
        plt.show()
    else:
        plt.close()

    return {
        "elbow_path": elbow_path,
        "dendrogram_path": dendrogram_path,
        "clusters_path": clusters_path,
    }


def main():
    args = parse_args()
    run_ward_clustering(
        matrices_dir=args.matrices_dir,
        output_dir=args.output_dir,
        max_k=args.max_k,
        n_clusters=args.n_clusters,
        show=args.show,
    )


if __name__ == "__main__":
    main()
