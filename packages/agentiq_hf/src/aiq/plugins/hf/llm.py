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
    model_d=config.model_dump(exclude={"type"}, by_alias=True)
    #print(Fore.RED+"model_d", model_d, Fore.RESET)
    model_name=model_d['model_name']
    model_type=model_d['model_type']
    device=model_d["device"]
    logger.debug("model_name is : %s", model_name)
    logger.debug("model_type is : %s", model_type)
    logger.debug("device used is : %s", device)
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"    
    # Initialize Tokenizer & Model
    if model_type=='AutoModelForCausalLM':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif model_type=="AutoModelForSequenceClassification":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        logger.debug("you should specify a model_type in config.yml , currently supported model_type are :AutoModelForCausalLM,AutoModelForMaskedLM,AutoModelForQuestionAnswering,AutoModelForSequenceClassification,")
    model.eval()
    model.to(device)
    yield tokenizer, model
