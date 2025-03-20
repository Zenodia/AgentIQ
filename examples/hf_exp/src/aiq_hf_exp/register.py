# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
logger = logging.getLogger(__name__)

class HFExpsWorkflowConfig(FunctionBaseConfig, name="hf_exp"):

    # Add settings
    llm_name: str
    model_name: str

@register_function(config_type=HFExpsWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def simple_workflow(config: HFExpsWorkflowConfig, builder: Builder):

    import os
    #from smolagents import  LiteLLMModel,  tool
    from smolagents.agents import CodeAgent, ToolCallingAgent
    import time
    from smolagents.models import OpenAIServerModel
    from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool,tool
    import litellm    
    from .text_to_world_tool import text_to_world
    # Enable verbose mode for debugging
    #litellm.set_verbose = True # depreciated warning, use the below line instead
    os.environ['LITELLM_LOG'] = 'DEBUG'    
    base_url = "https://integrate.api.nvidia.com/v1"
    model = LiteLLMModel(
        model_id="openai/nvdev/deepseek-ai/deepseek-r1",
        api_base= base_url, 
        api_key=os.environ["NVIDIA_API_KEY"], 
        num_ctx=8192,  
    )
    #agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, add_base_tools=False)
    agent = CodeAgent(
        tools=[text_to_world, DuckDuckGoSearchTool()],
        model=model,
        add_base_tools=True,
        verbosity_level=2
    )


    async def _response_fn(input_message: str) -> str:
        agent_output = agent.run(input_message)
        is_string=isinstance(agent_output, str)
        if is_string:
            agent_output=agent_output
        else:
            agent_output=str(agent_output)
        return agent_output
    yield _response_fn
