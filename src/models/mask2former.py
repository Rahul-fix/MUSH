from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import torch

def get_mask2former_model(num_labels, device, backbone="facebook/mask2former-swin-large-ade-semantic"):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        backbone,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    model.to(device)
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
