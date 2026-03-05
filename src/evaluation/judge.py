from openai import OpenAI


def create_qwen_client(api_key: str, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1") -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def run_qwen_judge(prompt: str, client: OpenAI, model="qwen-turbo") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        seed=123,
    )
    return response.choices[0].message.content
