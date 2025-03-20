# test run and see that you can genreate a respond successfully 
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import smolagents
from langchain import prompts, chat_models, hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import Optional, List
from operator import itemgetter
import requests
from dotenv import load_dotenv
load_dotenv()
import os , sys
import zipfile
from pathlib import Path
import os
from pathlib import Path
from smolagents import  LiteLLMModel,  tool
from colorama import Fore
api_key=os.environ["NVIDIA_API_KEY"]

llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
query_rewrite_prompt="""[INST]
You are an expert level writer who is able to take in a user input query and re-write it to a elaborated description for a short video clip. 
you will produce a descriptions that anchor how the resulting video clip should look like for promotion purposes.
<EXAMPLE>
user input : humanoid robot working in a chemical factory.
output: A first person view from the perspective from a human sized robot as it works in a chemical plant. The robot has many boxes and supplies nearby on the industrial shelves. The camera on moving forward, at a height of 1m above the floor. Photorealistic"
</EXAMPLE>

you will format the re-written output following STRICTLY the guideline below :
 - include only the re-written output only, do not use prefix it with something like 'Here is the elaborated description: ' nor suffix either.
 - do NOT include words such as 'the camera follows ...'
 - summarize into max 3 sentences and make it short and concise

You will return the re-written output from user_input : {user_input}

Begin !
[/INST]
"""
rewrite_prompt = PromptTemplate(
input_variables=['user_input'],
template=query_rewrite_prompt,
)
query_rewrite_chain = (rewrite_prompt | llm | {"output": StrOutputParser()})

video_out='C:\\Users\\zcharpy\\Contacts\\smolagents\\video_out\\'

def text_to_video(text):
    """based on the user query, modify the theme prompts, then create a video satisfying to the user query, at the end return the path to the generated mp4 video

    Args:
        user query: contain information about the theme of the video that user wishes to generate
    """
    invoke_url = "https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos-1.0-7b-diffusion-text2world"
    fetch_url_format = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
    #rewriting user input prompt to video generation prompt 
    
    prompt=query_rewrite_chain.invoke({"user_input":text})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    
    payload = {
      "inputs": [
        {
          "name": "command",
          "shape": [1],
          "datatype": "BYTES",
          "data": [
            f"text2world --prompt=\"{prompt}\""
          ]
        }
      ],
      "outputs": [
        {
          "name": "status",
          "datatype": "BYTES",
          "shape": [1]
        }
      ]
    }
    
    # re-use connections
    session = requests.Session()
    
    response = session.post(invoke_url, headers=headers, json=payload)
    
    while response.status_code == 202:
        request_id = response.headers.get("NVCF-REQID")
        fetch_url = fetch_url_format + request_id
        response = session.get(fetch_url, headers=headers)
    response = requests.post(invoke_url, headers=headers, json=payload)
    
    response.raise_for_status()
    
    with open(video_out+'result.zip', 'wb') as f:
      f.write(response.content)
    return video_out+'result.zip'


@tool
def text_to_world(user_query : str)-> str :
    """return the full path to the location of a generated video based on user_query in mp4 format
    Args:
        user_query: the user input
    """
    video_zip_file=text_to_video(user_query)
    print(Fore.CYAN +"generate and output video location", video_zip_file , Fore.RESET)

    return video_zip_file
