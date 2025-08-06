from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import torch

def get_mask2former_model(num_labels, device=None):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        # "facebook/mask2former-swin-tiny-ade-semantic",  # Swin-Tiny backbone for lowest memory
        # "facebook/mask2former-swin-large-ade-semantic",  # Swin-Large backbone for lowest memory
        "facebook/mask2former-swin-base-ade-semantic",  # Swin-Base backbone for lowest memory
        # "facebook/mask2former-swin-small-ade-semantic",  # Swin-Small backbone for lowest memory
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    # Do NOT enable gradient checkpointing (not supported)
    # model.to(device)
    return model

def get_preprocessor(num_labels):
    return Mask2FormerImageProcessor(
        ignore_index=255,
        reduce_labels=False,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        num_labels=num_labels
    )
