import numpy as np
import torch

def remap_labels(mask: np.ndarray, label2id_map: dict) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.int64)
    remapped_mask = torch.zeros_like(mask)
    for old_id, new_id in label2id_map.items():
        remapped_mask[mask == old_id] = new_id
    return remapped_mask

# Example palettes and label mappings
id2label = {
    0: "bg",
    11: "pepper_kp",
    12: "pepper_red",
    13: "pepper_yellow",
    14: "pepper_green",
    15: "pepper_mixed",
    17: "pepper_mixed_red",
    18: "pepper_mixed_yellow"
}
label2id = {old_id: new_id for new_id, old_id in enumerate(sorted(id2label.keys()))}
id2label_remapped = {new_id: id2label[old_id] for old_id, new_id in label2id.items()}
id2color = {
    0: "#000000",
    1: "#0000ff",
    2: "#c7211c",
    3: "#fff700",
    4: "#00ff00",
    5: "#e100ff",
    6: "#ff6600",
    7: "#d1c415",
}
palette = []
for class_id in range(len(id2label_remapped)):
    hex_color = id2color.get(class_id, "#000000")
    rgb = tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    palette.append(rgb)
palette = np.array(palette, dtype=np.uint8)
