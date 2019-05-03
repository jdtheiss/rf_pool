import torch
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import offsetbox

def confusion_score(i1, i2, labels, feature_vectors):
    """
    Computes the average cosign difference between 
    all pairs of feature vectors between class labels i1 and i2 
    
    Parameters
    ----------
    i1: int
        first class label
    i2: int
        second class label
        
    Returns
    -------
    score: torch.tensor
        the average cosign difference between all pairs 
        of feature vectors between class label i1 and i2
    """
    
    # get feature vectors
    features_i1 = feature_vectors[np.where(labels == i1)[0]]
    features_i2 = feature_vectors[np.where(labels == i2)[0]]
    
    # compute inner products
    inner = torch.matmul(features_i1, torch.t(features_i2))
    
    # calculate the norms
    norm_i1 = torch.sqrt(torch.diagonal(torch.matmul(features_i1, torch.t(features_i1))).reshape(1,-1))
    norm_i2 = torch.sqrt(torch.diagonal(torch.matmul(features_i2, torch.t(features_i2))).reshape(1,-1))
    norm = torch.matmul(torch.t(norm_i1), norm_i2)
    
    # avarage over the cosign differences 
    score = torch.mean(torch.divide(inner, norm)) 
    
    return score

def confusion_matrix(labels, feature_vectors):
    """
    Computes a confusion matrix from a set of feature vectors and corresponding class labels
    
    Parameters
    ----------
    labels: torch.tensor
        set of all class labels
    feature_vectors: torch.tensor
        set of all corresponding class feature vectors
        
    Returns
    -------
    matrix: torch.tensor
        confusion matrix of average cosign differences
    unique_labels: torch.tensor
       class labels corresponding to matrix entries
    """
    unique_labels = torch.unique(labels)
    matrix = torch.zeros((len(unique_labels,)*2)
    
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_label):
            matrix[i, j] = confusion_score(label_i, label_j, labels, feature_vectors)
    
    return matrix, unique_labels

def plot_confusion_matrix(m):
    raise NotImplemented
                         

def visualize_embedding(embeddings, images, labels=None, cmap='tab10', figsize=(15,15)):
    """
    2D Visualization for a dimensionally-reduced set of embeddings. 
    Plots the images at the embedding points 
    
    Paramaters
    ----------
    embeddings: numpy.array
        2-d point coordinates of feature vector embeddings
    images: torch.tensor
        input images
    labels: torch.tensor
        corresponding class labels

    Returns
    -------

    """
    images = images.permute(0,2,3,1)

    ax_max = np.max(embeddings, axis=0)
    ax_min = np.min(embeddings, axis=0)
    ax_dist = np.linalg.norm(ax_max - ax_min)
    min_dist = (1/(3*figsize[0])) * ax_dist


    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    cmap = plt.get_cmap(cmap)

    shown_embeddings = np.ones((1,2))
    for i in range(embeddings.shape[0]):
        dist_to_embeddings = np.linalg.norm((shown_embeddings - embeddings[i]), axis=1)

        if np.min(dist_to_embeddings) >= min_dist:
            shown_embeddings = np.vstack((shown_embeddings, embeddings[i,None]))
            patch = images[i]

            if labels is not None:
                if images.shape[-1] != 3:
                    patch = color_code_patch(patch.numpy(), labels[i].numpy(),cmap)

            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(patch, zoom=1, cmap=plt.cm.gray_r),
                xy=embeddings[i], frameon=False
                )
            ax.add_artist(imagebox)

    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.show()

def color_code_patch(patch, label, cmap):
    patch = np.tile(patch, (1,1,3))
    patch = (1,1,1) * (1 - patch) + cmap(label)[:3] * patch 
    return np.clip(patch, 0, 1)




