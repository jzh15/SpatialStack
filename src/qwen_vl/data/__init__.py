import re
from pathlib import Path


def _resolve_existing_path(primary_path, *fallback_paths):
    for candidate in (primary_path, *fallback_paths):
        if Path(candidate).exists():
            return candidate
    return primary_path

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

SPAR = {
    "annotation_path": _resolve_existing_path(
        "data/train/spar_7m.jsonl",
        "data/vgllm/train/spar_7m.jsonl",
    ),
    "data_path": "data/media",
    "tag": "3d"
}

SPAR_234K = {
    "annotation_path": _resolve_existing_path(
        "data/train/spar_234k.json",
        "data/vgllm/train/spar_234k.json",
    ),
    "data_path": "data/media",
    "tag": "3d"
}

LLAVA_HOUND = {
    "annotation_path": _resolve_existing_path(
        "data/train/llava_hound_255k.json",
        "data/vgllm/train/llava_hound_255k.json",
    ),
    "data_path": "data/media",
    "tag": "2d"
}

LLAVA_HOUND_64K = {
    "annotation_path": _resolve_existing_path(
        "data/train/llava_hound_64k.json",
        "data/vgllm/train/llava_hound_64k.json",
    ),
    "data_path": "data/media",
    "tag": "2d"
}

SCANNET_DET = {
    "annotation_path": _resolve_existing_path(
        "data/train/scannet_det_train_4frames.json",
        "data/vgllm/train/scannet_det_train_4frames.json",
    ),
    "data_path": "data/media",
    "tag": "3d"
}

SCANREFER = {
    "annotation_path": _resolve_existing_path(
        "data/train/scanrefer_train_32frames.json",
        "data/vgllm/train/scanrefer_train_32frames.json",
    ),
    "data_path": "data/media",
    "tag": "3d"
}

SCAN2CAP = {
    "annotation_path": _resolve_existing_path(
        "data/train/scan2cap_train_32frames.json",
        "data/vgllm/train/scan2cap_train_32frames.json",
    ),
    "data_path": "data/media",
    "tag": "3d"
}

VLM3R_SCANNET = {
    "annotation_path": "data/vlm3r/annotations/vsibench_train/merged_qa_scannet_train.json",
    "data_path": "data/vlm3r/media",
    "tag": "3d"
}

VLM3R_SCANNETPP = {
    "annotation_path": "data/vlm3r/annotations/vsibench_train/merged_qa_scannetpp_train.json",
    "data_path": "data/vlm3r/media",
    "tag": "3d"
}

VLM3R_ROUTEPLAN = {
    "annotation_path": "data/vlm3r/annotations/vsibench_train/merged_qa_route_plan_train.json",
    "data_path": "data/vlm3r/media",
    "tag": "3d"
}

VSI_APPR_ORDER = {
    "annotation_path": "data/vsi_590k/annotations/vsi_appearance_order_vsibench_scannet.json",
    "data_path": "data/vsi_590k/media",
    "tag": "3d"
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "spar": SPAR,
    "llava_hound": LLAVA_HOUND,
    "scannet_det": SCANNET_DET,
    "scanrefer": SCANREFER,
    "scan2cap": SCAN2CAP,
    "spar_234k": SPAR_234K,
    "llava_hound_64k": LLAVA_HOUND_64K,
    "vlm3r_scannet": VLM3R_SCANNET,
    "vsi_appr_order": VSI_APPR_ORDER,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config["dataset_name"] = dataset_name
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
