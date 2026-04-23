import logging
import os
import sys
from typing import List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("lmms-eval")


def _ensure_spatialbot_on_path() -> None:
    repo_path = os.environ.get("SPATIALBOT_PATH", "/workspace/SpatialBot")
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)


_ensure_spatialbot_on_path()

try:
    from bunny.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from bunny.conversation import SeparatorStyle, conv_templates
    from bunny.model.builder import load_pretrained_model
    from bunny.util.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
except Exception as e:
    eval_logger.debug(f"SpatialBot (bunny) is not available on PYTHONPATH. Error: {e}")


@register_model("spatialbot")
class SpatialBot(lmms):
    def __init__(
        self,
        pretrained: str = "RussRobin/SpatialBot-3B",
        model_base: Optional[str] = None,
        model_type: str = "phi-2",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[int] = 1,
        conv_template: str = "bunny",
        device_map: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self._device = torch.device(device)
        self.device_map = device_map or device

        self.pretrained = pretrained
        self.model_type = model_type
        self.model_name = get_model_name_from_path(pretrained)
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            pretrained,
            model_base,
            self.model_name,
            model_type,
            device_map=self.device_map,
        )
        self._config = self._model.config
        self.conv_template = conv_template
        self.batch_size_per_gpu = int(batch_size)

        self.model.eval()
        if device_map != "auto":
            self.model.to(self._device)

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("SpatialBot loglikelihood is not implemented for lmms_eval.")

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            if visuals in (None, [None]):
                images = None
            else:
                if not isinstance(visuals, list):
                    visuals = [visuals]
                images = visuals

            if images is None:
                qs = contexts
                image_tensor = None
            else:
                if not all(isinstance(img, Image.Image) for img in images):
                    raise NotImplementedError("SpatialBot only supports image inputs for now.")
                qs = DEFAULT_IMAGE_TOKEN + "\n" + contexts
                image_tensor = process_images(images, self._image_processor, self.model.config)

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
            input_ids = input_ids.to(self.device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 64
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=self.model.dtype, device=self.device, non_blocking=True)
                    if image_tensor is not None
                    else None,
                    do_sample=gen_kwargs.get("do_sample", False),
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            input_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)].strip()

            res.append(outputs)
            self.cache_hook.add_partial("generate_until", (contexts, gen_kwargs), outputs)
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("SpatialBot multi-round generation is not implemented.")
