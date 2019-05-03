import torch
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import offsetbox


def color_code_patch(patch, label, cmap):
    patch = np.tile(patch, (1,1,3))
    patch = (1,1,1) * (1 - patch) + cmap(label)[:3] * patch 
    return np.clip(patch, 0, 1)

def visualize_embedding(embedding, inputs, labels=None, cmap='tab10', figsize=(15,15)):
    """
    Paramaters
    ----------
    embedding:
    inputs:
    labels:

    Returns
    -------

    """
    inputs = inputs.permute(0,2,3,1)

    ax_max = np.max(embedding, axis=0)
    ax_min = np.min(embedding, axis=0)
    ax_dist = np.linalg.norm(ax_max - ax_min)
    min_dist = (1/(3*figsize[0])) * ax_dist


    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    cmap = plt.get_cmap(cmap)

    shown_embeddings = np.ones((1,2))
    for i in range(embedding.shape[0]):
        dist_to_embeddings = np.linalg.norm((shown_embeddings - embedding[i]), axis=1)

        if np.min(dist_to_embeddings) >= min_dist:
            shown_embeddings = np.vstack((shown_embeddings, embedding[i,None]))
            patch = inputs[i]

            if labels is not None:
                if inputs.shape[-1] != 3:
                    patch = color_code_patch(patch.numpy(), labels[i].numpy(),cmap)

            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(patch, zoom=1, cmap=plt.cm.gray_r),
                xy=embedding[i], frameon=False
                )
            ax.add_artist(imagebox)

    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.show()







