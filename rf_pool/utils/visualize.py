import torch
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import offsetbox

def rf_interference(feat_i, feat_j):
    """
    Computes a receptive field interference score:
    mena cosine similarity between feat_i and rf_pool(feat_i, feat_j)
    
    Parameters
    ----------
    feat_i: torch.tensor
        features corresponding to first class label
    feat_j: torch.tensor
        features corresponding to second class label
        
    Returns
    -------
    mean: torch.tensor
        the average cosine similarity 
    std: torch.tensor
        std dev of cosine similarities 
    """
    n_i = feat_i.shape[0]
    n_j = feat_j.shape[0]
    feat_i_r = torch.unsqueeze(feat_i, -1).repeat(1,1,n_j)
    feat_j_r = torch.unsqueeze(feat_j, -1).repeat(1,1,n_i)
    
    # max across channels
    feat_p = torch.max(feat_i_r, feat_j_r.permute(2,1,0))
    
    # inner product
    inner = torch.sum(feat_i_r * feat_p, dim=1)
    
    # norms
    norm_i = torch.sqrt(torch.diagonal(torch.matmul(feat_i, torch.t(feat_i))))
    norm_i = torch.unsqueeze(norm_i, -1).repeat(1,n_j)
    norm_j = torch.sqrt(torch.sum(torch.pow(feat_p, 2), dim=1))
    norm = norm_i * norm_j
    
    # average over cosine similarity
    cosine_sim = inner / norm
    
    return 1 - torch.mean(cosine_sim), torch.std(cosine_sim)

def pairwise_cosine_similarity(feat_i, feat_j):
    """
    Computes pair-wise cosine similarity
    between features in feat_i and feat_j
    
    Parameters
    ----------
    feat_i: torch.tensor
        features corresponding to class label i
    feat_j: torch.tensor
        features corresponding to class label j
        
    Returns
    -------
    mean: torch.tensor
        the average cosine similarity 
    std: torch.tensor
        std dev of cosine similarities 
    """

    inner = torch.sum(torch.mul(feat_i, feat_j), dim=1)
    norm_i = torch.sqrt(torch.sum(torch.pow(feat_i, 2), dim=1))
    norm_j = torch.sqrt(torch.sum(torch.pow(feat_j, 2), dim=1))
    
    cosine_sim = torch.div(inner, torch.mul(norm_i, norm_j))
    
    return torch.mean(cosine_sim), torch.std(cosine_sim)
    
def cosine_similarity(feat_i, feat_j):
    """
    Computes the mean cosine similarity across
    every permutation pair of the features 
    in feat_i and feat_j
    
    Parameters
    ----------
    feat_i: torch.tensor
        features corresponding to class label i
    feat_j: torch.tensor
        features corresponding to class label j
        
    Returns
    -------
    mean: torch.tensor
        the average cosine similarity 
    std: torch.tensor
        std dev of cosine similarities 
    """
    # compute inner products
    inner = torch.matmul(feat_i, torch.t(feat_j))
    
    # calculate the norms
    norm_i = torch.sqrt(torch.diagonal(torch.matmul(feat_i, torch.t(feat_i)))).reshape(1,-1)
    norm_j = torch.sqrt(torch.diagonal(torch.matmul(feat_j, torch.t(feat_j)))).reshape(1,-1)
    norm = torch.matmul(torch.t(norm_i), norm_j)
    
    # avarage over the cosine similarity 
    cosine_sim = inner/norm 
    
    return torch.mean(cosine_sim), torch.std(cosine_sim)

def confusion_matrix(feature_vectors, labels, interference_fn=cosine_similarity):
    """
    Computes a mean cosine similarity matrix
    from a set of feature vectors and corresponding class labels
    
    Parameters
    ----------
    feature_vectors: torch.tensor
        set of all corresponding class feature vectors
    labels: torch.tensor
        set of all class labels
    interference_fn: visualize.fn
        computes an interference score between features
    Returns
    -------
    matrix: torch.tensor
        confusion matrix of mean rf-interference scores
    unique_labels: torch.tensor
       class labels corresponding to matrix entries
    """
    unique_labels = torch.unique(labels, sorted=True)
    mean_matrix = torch.zeros((len(unique_labels),)*2)
    std_matrix = torch.zeros((len(unique_labels),)*2)
    
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            feat_i = feature_vectors[np.where(labels == label_i)[0]]
            feat_j = feature_vectors[np.where(labels == label_j)[0]] 
            # compute the mean cosine similarity cosine
            mean_matrix[i, j], std_matrix[i,j]  = interference_fn(feat_i, feat_j)
    
    return mean_matrix, std_matrix, unique_labels
    
def show_confusion_matrix(data, labels, cmap=plt.cm.afmhot):
    """
    TODO
    """
    fig, ax = plt.subplots(1,1)
    heatmap = ax.pcolor(data, cmap=cmap)
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
    
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(labels.numpy(), minor=False)
    ax.set_yticklabels(labels.numpy(), minor=False) 
    fig.colorbar(heatmap)

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
            
            # colorize images
            if labels is not None:
                if images.shape[-1] != 3:
                    patch = color_code_patch(patch.numpy(), labels[i].numpy(),cmap)
            else:
                patch = torch.flatten(patch, -2)

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




