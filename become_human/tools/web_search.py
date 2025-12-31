from typing import Annotated
import os
import aiohttp
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

baidu_api_key = os.getenv('QIANFAN_API_KEY')
dashscope_api_key = os.getenv('DASHSCOPE_API_KEY')
dashscope_api_base = os.getenv('DASHSCOPE_API_BASE')
dashscope_search_model = os.getenv('DASHSCOPE_SEARCH_MODEL_NAME')

# 这个工具名似乎与openai/anthropic等厂商的内置工具名冲突？如果将此工具设置为tool_choice可能会有影响
@tool
async def web_search(query: Annotated[str, '使用自然语言的搜索语句']) -> str:
    """使用网络获取信息。这适用于获取未知的信息，或是为了确认信息真实可靠，尤其适合获取那些具有强时效性的信息。"""
    # 尝试使用百度搜索
    if baidu_api_key:
        url = 'https://qianfan.baidubce.com/v2/ai_search/chat/completions'
        headers = {
            'Authorization': f'Bearer {baidu_api_key}',
            'Content-Type': 'application/json'
        }
        messages = [
            {
                "content": query,
                "role": "user"
            }
        ]
        data = {
            "messages": messages,
            "search_source": "baidu_search_v2",
            "resource_type_filter": [{"type": "web","top_k": 7}],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=20)) as response:
                if response.status == 200:
                    response_json = await response.json()
                    references = response_json["references"]
                    parsed_references = '以下是搜索到的网页信息：\n\n' + '\n\n'.join([f'- date: {reference["date"]}, title: {reference["title"]}, content: {reference["content"]}' for reference in references])
                    return parsed_references
                else:
                    error_text = await response.text()
                    raise Exception(f"网页搜索API请求失败，状态码: {response.status}, 错误信息: {error_text}")

    # 如果没有百度API密钥，尝试使用阿里云百炼平台
    if dashscope_api_key and dashscope_search_model:
        # 创建具有网络搜索功能的代理
        llm = ChatOpenAI(
            model=dashscope_search_model,
            base_url=dashscope_api_base,
            api_key=dashscope_api_key,
            extra_body={
                "enable_search": True,
                "search_options": {
                    "search_strategy": "turbo", # 默认turbo，还有max和agent
                    "forced_search": True,
                    "enable_search_extension": True, # 垂类数据，如天气可直接通过专门的api查询
            }},
        )

        # 创建一个简单的代理来执行搜索（也许以后可以添加一个工具使agent可以多轮查询，目前只能单轮）
        agent = create_agent(
            model=llm,
            system_prompt="""你是一个信息获取工具，你面对的用户实际上是一个AI Agent。
请返回用户想要了解的，详尽并保持可读的有用的信息，保证信息真实可靠以及可能的时效性问题，但不要过于冗杂导致挤占用户的上下文。
也不要提供根据信息产生的推理结果，如建议、提醒等，请专注于提供信息本身。""",
        )

        # 调用代理并启用搜索功能
        response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})

        # 提取响应内容
        if isinstance(response, dict) and "messages" in response:
            last_message = response["messages"][-1]
            if hasattr(last_message, "content"):
                return getattr(last_message, "content")
            elif isinstance(last_message, dict) and "content" in last_message:
                return last_message['content']

        return str(response)

    # 如果都没有配置API密钥
    raise ValueError("系统未设置环境变量QIANFAN_API_KEY或DASHSCOPE_API_KEY和DASHSCOPE_SEARCH_MODEL_NAME，无法使用此工具，请暂时跳过此工具。")
