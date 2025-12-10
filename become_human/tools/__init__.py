import os
#from langchain_sandbox import PyodideSandboxTool
from become_human.tools.send_message import send_message
from become_human.tools.retrieve_memories import retrieve_memories
from become_human.tools.add_self_call import add_self_call
from become_human.tools.web_search import web_search

CORE_TOOLS = [
    send_message,
    retrieve_memories,
    add_self_call,
]
if os.getenv('QIANFAN_API_KEY') or (os.getenv('DASHSCOPE_API_KEY') and os.getenv('DASHSCOPE_SEARCH_MODEL_NAME')):
    CORE_TOOLS.append(web_search)
# CORE_TOOLS.append(PyodideSandboxTool(description='''一个安全的 Python 代码沙盒，使用此沙盒来执行 Python 命令，特别适合用于数学计算。
# - 输入应该是有效的 Python 命令。
# - 要返回输出，你应该使用print(...)将其打印出来。
# - 打印输出时不要使用 f 字符串。
# 注意：
# - 沙盒没有连接网络。
# - 沙盒是无状态的，变量不会被继承到下一次调用。'''))
