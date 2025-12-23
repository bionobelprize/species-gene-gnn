import argparse
import torch 

def main():
    parser = argparse.ArgumentParser(description="PCA analysis for species/gene embeddings.")
    parser.add_argument('--model', type=str, required=True, help='Path to the model state file')
    parser.add_argument('--embedding', type=str, choices=['species', 'gene'], default='species', help='Embedding type to analyze')
    parser.add_argument('--device', type=str, default='cpu', help='Device to load model (cpu or cuda)')
    parser.add_argument('--n_components', type=int, default=2, help='Number of PCA components (2 or 3)')
    parser.add_argument('--output', type=str, default=None, help='Path to save the PCA plot (optional)')
    args = parser.parse_args()

    device = torch.device(args.device)
    data = load_embedding_from_model_state(args.model, device)
    if args.embedding == 'species':
        embeddings = data['species_embeddings']
        id_map = data['id_to_species']
        title = 'Species Embedding PCA'
    else:
        embeddings = data['gene_embeddings']
        id_map = data['id_to_gene']
        title = 'Gene Embedding PCA'

    reduced = perform_pca(embeddings, n_components=args.n_components)
    plot_pca_results(reduced, id_map=id_map, title=title, save_path=args.output)


def plot_pca_results(reduced_embeddings, id_map=None, title="PCA Result", save_path=None):
    """
    Visualize PCA-reduced embeddings.
    Args:
        reduced_embeddings (np.ndarray): Embeddings after PCA (n_samples, n_components).
        id_map (dict): Mapping from index to id string (species or gene).
        title (str): Plot title.
        save_path (str): If provided, save the plot to this path.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    n_components = reduced_embeddings.shape[1]
    plt.figure(figsize=(8, 6))
    if n_components == 2:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
        if id_map:
            for i, txt in id_map.items():
                plt.annotate(txt, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.7)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.axes(projection='3d')
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], alpha=0.7)
        if id_map:
            for i, txt in id_map.items():
                ax.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], reduced_embeddings[i, 2], txt, fontsize=8, alpha=0.7)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
    else:
        raise ValueError("Only 2D or 3D PCA visualization is supported.")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def load_embedding_from_model_state(path: str, device: torch.device) -> dict:
    """
    Load a model state dictionary from a given path.
    
    """
    state_dict = torch.load(path, map_location=device)
    species_embeddings = state_dict['model_state_dict']['species_embedding.weight']
    gene_embeddings = state_dict['model_state_dict']['gene_embedding.weight']
    id_to_species = state_dict.get('id_to_species', None)
    id_to_gene = state_dict.get('id_to_gene', None)
    return {
        'species_embeddings': species_embeddings,
        'gene_embeddings': gene_embeddings,
        'id_to_species': id_to_species,
        'id_to_gene': id_to_gene
    }

# PCA dimensionality reduction function
def perform_pca(embeddings, n_components=2):
    """
    Perform PCA on the given embeddings and reduce to n_components dimensions.
    Args:
        embeddings (Tensor or np.ndarray): High-dimensional embeddings.
        n_components (int): Number of dimensions to reduce to.
    Returns:
        np.ndarray: Embeddings reduced to n_components dimensions.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    if hasattr(embeddings, 'detach'):
        embeddings = embeddings.detach().cpu().numpy()
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    return reduced

if __name__ == "__main__":
    main()