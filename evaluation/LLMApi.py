import time

from openai import OpenAI
import tiktoken

API_BASE = ""
API_KEY = ""


class llmApi():
    def __init__(self, model="gpt-4o-2024-08-06", api_key="", api_base=""):
        if api_base or API_BASE:
            self.client = OpenAI(
                api_key=api_key if api_key else API_KEY,
                base_url=api_base if api_base else API_BASE,
                )
        else:
            self.client = OpenAI(api_key=API_KEY)
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model('gpt-4o-2024-08-06')

    def get(self, prompt, max_new_tokens=512, max_tries=3):
        for i in range(max_tries):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            }
                        ],
                    model=self.model,
                    max_tokens=max_new_tokens,
                    timeout=60
                    )
                return response.choices[0].message.content
            except Exception as e:
                time.sleep(30)
                print(f"API Error: {e}, retry: {i + 1}")
        return ""

    def get_by_stream(self, prompt, max_new_tokens=512, max_tries=3):
        for i in range(max_tries):
            try:
                response = self.client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": prompt,
                        }],
                    model=self.model,
                    max_tokens=max_new_tokens,
                    stream=True,  # 启用流式接收
                    timeout=60
                    )

                # 逐步接收并构建完整的响应内容
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        message = chunk.choices[0].delta.content
                        full_response += message

                return full_response

            except Exception as e:
                time.sleep(30)
                print(f"API Error: {e}, retry: {i + 1}")
        return ""


def api_get(prompt, model="gpt-4o-2024-05-13", maxtries=3):
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
        )
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
                }
            ],
        model=model,
        )
    return response.choices[0].message.content

