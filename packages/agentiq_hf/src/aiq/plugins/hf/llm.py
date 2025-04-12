# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_llm_client
from aiq.llm.nim_llm import NIMModelConfig
from aiq.llm.openai_llm import OpenAIModelConfig
from aiq.llm.huggingface_llm import HuggingFaceModelConfig
#from transformers import LlamaForCausalLM, GenerationConfig, pipeline
#from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch
from colorama import Fore
import torch
from transformers import (pipeline, AutoTokenizer, AutoModel,AutoProcessor,
AutoModelForCausalLM,
AutoModelForMaskedLM,
AutoModelForQuestionAnswering,
AutoModelForSequenceClassification)

def load_model(model_name, device, num_gpus, load_8bit=False, debug=False):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: "13GiB" for i in range(num_gpus)},
                })
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        #replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    if "Llama-4" in model_name:        
        processor = AutoProcessor.from_pretrained(model_id)
        model = Llama4ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="flex_attention",
            torch_dtype=torch.bfloat16,
            **kwargs
        )

    elif 'chatglm' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    elif 'moonshot' in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,                    
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cuda:0",            
        )
        tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, device_map="cuda:0")

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, **kwargs)

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer
#from colorama import Fore
logger = logging.getLogger(__name__)


@register_llm_client(
    config_type=HuggingFaceModelConfig,
    wrapper_type=LLMFrameworkEnum.HF
)
async def register_huggingface_client(config: HuggingFaceModelConfig, builder: Builder):
    """Register HuggingFace LLM client."""
    import torch
    from transformers import (pipeline, AutoTokenizer, AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification)
    
    # Initialize Variables
    model_d=config.model_dump()
    #print(Fore.RED+"model_d", model_d, Fore.RESET)
    model_name=model_d['model_name']
    model_type=model_d['model_type']
    print(Fore.RED + "model dictionary", model_d, '\n\n',   Fore.RESET)
    device=model_d["device"]
    num_gpus=torch.cuda.device_count()
    logger.debug("model_name is : %s", model_name)
    logger.debug("model_type is : %s", model_type)
    logger.debug("device used is : %s", device)
    #logger.debug("number of available GPUs : %s", num_gpus)
    
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"    
    # Initialize Tokenizer & Model
    if model_type=='AutoModelForCausalLM':
        #tokenizer = AutoTokenizer.from_pretrained(model_name,torch_dtype="auto",device_map="auto",trust_remote_code=True,)
        #model = AutoModelForCausalLM.from_pretrained(model_name)
        model, tokenizer=load_model(model_name, device, num_gpus, load_8bit=False, debug=False)
    elif model_type=="AutoModelForSequenceClassification":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        logger.debug("you should specify a model_type in config.yml , currently supported model_type are :AutoModelForCausalLM,AutoModelForMaskedLM,AutoModelForQuestionAnswering,AutoModelForSequenceClassification,")
    model.eval()
    model.to(device)
    yield tokenizer, model
