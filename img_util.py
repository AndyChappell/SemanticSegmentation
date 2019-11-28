import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def imagify(inputs, predictions, masks, void_code, n=3, randomize=True):
    """Process input, prediction and mask data ready for display
    Args:
        inputs (np.ndarray): Input tensor from a batch.
        predictions (np.ndarray): Predictions tensor from a batch.
        masks (np.ndarray): Mask tensor from a batch.
        void_code (int): The null mask code (typically zero).
        n (int): The number of images to extract from the batch.
        randomize (boolean): If True choose a random set of images from the
            batch, otherwise pick the first n.
    Returns:
        A zip of the three processed images ready for display.
    """
    # Select the images to process
    if randomize:
        choices = np.random.choice(np.array(range(inputs.shape[0])), size=n)
    else:
        choices = np.array(range(n))
    # Subset the inputs and masks
    input_imgs = inputs[choices,0,...]
    mask_imgs = masks[choices,...]

    # Create a void code mask, determine the class of each predicted pixel and
    # then apply the mask to remove non-hit regions
    msks = mask_imgs == void_code
    pred_imgs = np.argmax(predictions[choices,...], axis=1)
    pred_imgs = np.ma.array(pred_imgs, mask = msks).filled(0)
    if n > 1:
        return zip(input_imgs, pred_imgs, mask_imgs)
    else:
        return input_imgs, pred_imgs, mask_imgs
    #return zip(pred_imgs, mask_imgs)

def show_batch(epoch, batch, inputs, predictions, masks, void_code, n=3, randomize=True):
    """Display the images for a given epoch and batch. Each row is a triplet of
        input, prediction and mask.

    Args:
        epoch (int): The current training epoch.
        batch (int): The current training batch.
        inputs (np.ndarray): Input tensor from a batch.
        predictions (np.ndarray): Predictions tensor from a batch.
        masks (np.ndarray): Mask tensor from a batch.
        void_code (int): The null mask code (typically zero).
        n (int): The number of images to extract from the batch.
        randomize (boolean): If True choose a random set of images from the
            batch, otherwise pick the first n.
    """
    ax = None
    rows, cols = n, 3
    size = 6#10
    row_fac = 208. / 512
    col_fac = 1.
    if ax is None:
        fig, axs = plt.subplots(rows, cols, figsize=(cols * size * col_fac,
                                                    rows * size * row_fac))
    if rows == 1 and cols == 1:
        axs = [[axs]]
    elif (rows == 1 and cols != 1) or (rows != 1 and cols == 1):
        axs = [axs]
    axs = np.array(axs)

    bounds = np.linspace(0, 2, 3)
    norm = BoundaryNorm(bounds, cm.N)

    cmap = ListedColormap(['black', 'red', 'blue'])
    norm = BoundaryNorm([0., 0.5, 1.5, 2.], cmap.N)

    #xtr = dict(cmap="viridis", alpha=1.0)
    xtr = dict(cmap=cmap, norm=norm, alpha=1.0)

    images = imagify(inputs, predictions, masks, void_code, n, randomize)

    for img_triplet, ax_row in zip(images, axs):
        for img, ax in zip(img_triplet, ax_row):
            ax.imshow(img, **xtr)
    for ax in axs.flatten():
        ax.axis('off')
    plt.tight_layout()
    save_figure(plt, "diagnostic_{}_{}".format(epoch, batch))

def save_figure(fig, name):
    """Output a matplotlib figure PNG, PDF and EPS formats.

    Args:
        fig (Figure): The matplotlib figure to save.
        name (str): The output filename excluding extension.
    """
    fig.savefig(name + ".png")
    fig.savefig(name + ".pdf")
    fig.savefig(name + ".eps")

def get_supported_formats():
    """Output a matplotlib figure PNG, PDF and EPS formats.

    Args:

    Returns:
        A dictionary containing strings of file format descriptions keyed by
            extension.
    """
    return plt.gcf().canvas.get_supported_filetypes()
