# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import decord # must import decord after torch and before torchvision
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwen_vl.train.trainer
import qwen_vl.train.sampler
from qwen_vl.train.trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
)
from qwen_vl.data.data_qwen import make_supervised_data_module

from qwen_vl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer, AutoConfig, set_seed, enable_full_determinism
from transformers.utils.hub import cached_file

local_rank = None
QWEN3_5_MODEL_TYPES = {"qwen3_5", "qwen3_5_vl"}

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def resolve_model_modules(model):
    if hasattr(model, "visual") and hasattr(model, "model"):
        return model.visual, getattr(model.visual, "merger", None), model.model, model.lm_head

    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "visual") and hasattr(inner_model, "language_model"):
        return inner_model.visual, getattr(inner_model.visual, "merger", None), inner_model.language_model, model.lm_head

    raise ValueError(f"Unsupported model structure for training: {type(model)}")


def set_model(model_args, model):
    visual_module, merger_module, language_module, lm_head = resolve_model_modules(model)

    if model_args.tune_mm_vision:
        for n, p in visual_module.named_parameters():
            p.requires_grad = True
    else:
        for n, p in visual_module.named_parameters():
            p.requires_grad = False

    if merger_module is not None:
        if model_args.tune_mm_mlp:
            for n, p in merger_module.named_parameters():
                p.requires_grad = True
        else:
            for n, p in merger_module.named_parameters():
                p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in language_module.named_parameters():
            p.requires_grad = True
        for p in lm_head.parameters():
            p.requires_grad = True
    else:
        for n, p in language_module.named_parameters():
            p.requires_grad = False
        for p in lm_head.parameters():
            p.requires_grad = False

    if model_args.use_geometry_encoder:
        # vggt is frozen
        for n, p in model.geometry_encoder.named_parameters():
            p.requires_grad = False

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    # enable_full_determinism(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model_type = getattr(config, "model_type", None)

    if model_type == "qwen2_5_vl" or "qwen2.5" in model_args.model_name_or_path.lower():
        if not model_args.use_geometry_encoder:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        else:
            from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
            if hasattr(config, "use_geometry_encoder") and config.use_geometry_encoder != model_args.use_geometry_encoder:
                raise ValueError(
                    "The use_geometry_encoder in config and model_args are not consistent. "
                    "Please check the model config."
                )

            for k in [
                "use_geometry_encoder", 
                "geometry_encoder_type", 
                "reference_frame",
                "feature_fusion_method", 
                "fusion_num_layers",
                "geometry_merger_type",
                "geometry_fusion_layers",
                "geometry_encoder_layers",
                "include_camera_token",
                "pos_encoding_type",
                "vision_language_fusion_layers",
            ]:
                setattr(config, k, getattr(model_args, k))

            assert model_args.geometry_encoder_path is not None, \
                "geometry_encoder_path must be set in the config when use_geometry_encoder is True."
            model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                geometry_encoder_path=model_args.geometry_encoder_path
            )

        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    elif model_type == "qwen2_vl":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"
        data_args.processor = None
    elif model_type in QWEN3_5_MODEL_TYPES or "qwen3.5" in model_args.model_name_or_path.lower():
        if data_args.data_flatten:
            raise NotImplementedError(
                "Qwen3.5 training does not support data_flatten in this branch."
            )

        from transformers import Qwen3_5ForConditionalGeneration

        if model_args.use_geometry_encoder:
            from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry

            for k in [
                "use_geometry_encoder",
                "geometry_encoder_type",
                "geometry_encoder_path",
                "reference_frame",
                "feature_fusion_method",
                "fusion_num_layers",
                "geometry_merger_type",
                "geometry_fusion_layers",
                "geometry_encoder_layers",
                "include_camera_token",
                "pos_encoding_type",
                "vision_language_fusion_layers",
            ]:
                setattr(config, k, getattr(model_args, k))

            assert model_args.geometry_encoder_path is not None, (
                "geometry_encoder_path must be set in the config when use_geometry_encoder is True."
            )
            model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                geometry_encoder_path=model_args.geometry_encoder_path,
            )
        else:
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            padding_side="right",
        )
        data_args.image_processor = processor.image_processor
        data_args.processor = processor
        data_args.model_type = "qwen3.5"
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}' for training path: {model_args.model_name_or_path}"
        )

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    import torch.distributed as dist

    def is_rank_zero():
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    if is_rank_zero():
        visual_module, _, language_module, _ = resolve_model_modules(model)
        visual_module.print_trainable_parameters()
        language_module.print_trainable_parameters()

    print(model.config)
    if model_args.use_geometry_encoder:
        setattr(data_args, "use_geometry_encoder", model_args.use_geometry_encoder)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    if getattr(data_args, "processor", None) is not None:
        data_args.processor.save_pretrained(training_args.output_dir)
    else:
        data_args.image_processor.save_pretrained(training_args.output_dir)

    template_filename = "chat_template.json"
    template_path = os.path.join(training_args.output_dir, template_filename)

    source_path = None
    if os.path.isdir(model_args.model_name_or_path):
        candidate_path = os.path.join(model_args.model_name_or_path, template_filename)
        if os.path.isfile(candidate_path):
            source_path = candidate_path
    else:
        try:
            source_path = cached_file(
                model_args.model_name_or_path,
                template_filename,
                cache_dir=training_args.cache_dir,
            )
        except (OSError, EnvironmentError) as err:
            if getattr(data_args, "processor", None) is None:
                logging.warning("Unable to locate %s for model %s: %s", template_filename, model_args.model_name_or_path, err)

    if source_path:
        if os.path.abspath(source_path) != os.path.abspath(template_path):
            shutil.copy2(source_path, template_path)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
