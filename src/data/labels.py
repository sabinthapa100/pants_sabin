"""PanTS anatomical label definitions."""

CLASS_MAP = {
    1: "adrenal_gland_left",
    2: "adrenal_gland_right",
    3: "aorta",
    4: "bladder",
    5: "celiac_artery",
    6: "colon",
    7: "common_bile_duct",
    8: "duodenum",
    9: "femur_left",
    10: "femur_right",
    11: "gall_bladder",
    12: "kidney_left",
    13: "kidney_right",
    14: "liver",
    15: "lung_left",
    16: "lung_right",
    17: "pancreas",
    18: "pancreas_body",
    19: "pancreas_head",
    20: "pancreas_tail",
    21: "pancreatic_duct",
    22: "postcava",
    23: "prostate",
    24: "spleen",
    25: "stomach",
    26: "superior_mesenteric_artery",
    27: "veins",
    28: "pancreatic_lesion",
}

NAME_TO_CLASS = {
    name: class_id
    for class_id, name in CLASS_MAP.items()
}

BACKGROUND = 0

PANCREAS = 17
PANCREAS_BODY = 18
PANCREAS_HEAD = 19
PANCREAS_TAIL = 20
PANCREATIC_DUCT = 21
PANCREATIC_LESION = 28

PANCREAS_FAMILY = (
    PANCREAS,
    PANCREAS_BODY,
    PANCREAS_HEAD,
    PANCREAS_TAIL,
)
