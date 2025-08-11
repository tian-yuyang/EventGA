import openai
from openai import OpenAI, AzureOpenAI
import logging
import os
import time
from functools import cache
from anthropic import AnthropicBedrock
import httpx
# from sentence_transformers import SentenceTransformer

http_client = httpx.Client(proxies="Your proxy")
openai_api_key = "Your api key"
openai_api_base = "Your endpoint"
# openai_api_type = 'azure'
openai_api_version = 'api version'
openai_api_deploy = 'gpt-4o'
claude_api_deploy = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'
embeddings_model_name = 'text-embedding-ada-002'
# openai.api_key = openai_api_key
# openai.azure_endpoint = openai_api_base
# openai.api_version = openai_api_version
# openai.api_type = openai_api_type
Embedding_client = AzureOpenAI(
    api_version=openai_api_version,
    azure_endpoint=openai_api_base,
    api_key=openai_api_key,
    # http_client=http_client,
)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
class OpenAILLM:

    def __init__(self, llm_model_name, embedding_model_name):
        self.client = AzureOpenAI(
            api_version=openai_api_version,
            azure_endpoint=openai_api_base,
            api_key=openai_api_key,
            # http_client=http_client,
        )

        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)


    def get_llm_response(self, prompt, max_tokens=1024, timeout=10):
        # print("llm\n\n\n")
        # print(prompt)
        start = time.time()
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        n_retries = 10
        for i in range(n_retries):
            try:
                time.sleep(0.8*i)
                chat_completion = self.client.chat.completions.create(model=openai_api_deploy, messages=[{"role": "user", "content": prompt}], 
                                                                 max_tokens=max_tokens, timeout=timeout, temperature=0.7)
                end = time.time()
                # logging.log(logging.INFO, f"OpenAI response time: {end - start}")
                print(f"OpenAI response time: {end - start}")
                return chat_completion.choices[0].message.content
            except openai.APIError:
                print("openai.error.ServiceUnavailableError")
                pass
            except openai.APITimeoutError:
                print("openai.error.Timeout")
                pass
            except openai.APIError:
                print("openai.error.APIError")
                pass
            except openai.APIConnectionError:
                pass
            except openai.RateLimitError:
                print("openai.error.RateLimitError")
                time.sleep(10)
            # too many tokens
            except openai.BadRequestError:
                context_window = 3000 * 4 # max length in chars (every token is around 4 chars)
                prompt = prompt[:context_window]
                print("openai.error.InvalidRequestError")

        raise ValueError(f"OpenAI remains uncontactable after {n_retries} retries due to either Server Error or Timeout after {timeout} seconds")

    @cache
    def get_embeddings(self, query):
        # print("Embedding\n\n\n")
        # print(query)
        response = Embedding_client.embeddings.create(
            input=query,
            model=embeddings_model_name
        )
        embeddings = response.data[0].embedding
        return embeddings
    
class ClaudeLLM:

    def __init__(self, llm_model_name, embedding_model_name):
        
        self.client = AnthropicBedrock(
            aws_region="us-east-1",
            aws_access_key="aws_access_key",
            aws_secret_key="aws_secret_key",
            http_client=http_client,
        )
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)


    def get_llm_response(self, prompt, max_tokens=1024, timeout=10):
        # print(prompt)
        start = time.time()
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        n_retries = 10
        for i in range(n_retries):
            try:
                time.sleep(0.8*i)
                chat_completion = self.client.messages.create(model=claude_api_deploy, messages=[{"role": "user", "content": prompt}], 
                                                                 max_tokens=max_tokens, timeout=timeout, temperature=0.7)
                end = time.time()
                # logging.log(logging.INFO, f"OpenAI response time: {end - start}")
                # print(chat_completion)
                print(f"Claude response time: {end - start}")
                return chat_completion.content[0].text
            # too many tokens
            except Exception as e:
                print(f"Claude Error {e}")

        raise ValueError(f"Claude remains uncontactable after {n_retries} retries due to either Server Error or Timeout after {timeout} seconds")

    @cache
    def get_embeddings(self, query):
        response = Embedding_client.embeddings.create(
            input=query,
            model=self.embedding_model_name
        )
        embeddings = response.data[0].embedding
        return embeddings

