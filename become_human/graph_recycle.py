from typing import Literal, Sequence, Dict, Any, Union, Callable, Optional

#from typing_extensions import Annotated, TypedDict, Literal
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage, AnyMessage, RemoveMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
#from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig

from pydantic import BaseModel, Field

#from trustcall import create_extractor

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

#from langchain_community.embeddings import DashScopeEmbeddings
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from langchain_core.language_models.chat_models import BaseChatModel

from become_human.graph_base import BaseGraph
from become_human.memory import MemoryManager
from become_human.utils import parse_message, parse_messages
from become_human.config import get_thread_recycle_config
from become_human.time import now_seconds, AgentTimeSettings, real_time_to_agent_time, datetime_to_seconds

from langchain_text_splitters import RecursiveCharacterTextSplitter

import random

class MemoryEntry(BaseModel):
    content: str = Field(description="记忆的内容")
    stable_time: float = Field(description="记忆的稳定市场，单位为秒", ge=0.0)
    type: Literal["original", "summary", "semantic"] = Field(description="记忆的类型")
    creation_time_seconds: float = Field(description="记忆的创建时间总秒数")
    id: str = Field(description="记忆的唯一标识符")

class ExtractedMemoryInfo(BaseModel):
    '''使用这个工具来返回你从记录中提取到的记忆信息，请以第一人称视角提取记忆信息，提取到的所有记忆信息需满足以下规则：
0. 这份记录实际上来自一个AI，但请把它当作人类来撰写你提取到的信息，不要暴露这个人是一个AI的事实
1. 代词转换规则：
   - 在你最终输出的信息里出现的"我"只能指代当前用户（即需要提取记忆信息的主体），这是为了保持第一人称视角
   - 除"我"外，所有代词（你/他/她/他们）必须替换为完整姓名
   - 例：原句"他昨天去了超市" → "李四在2023-04-05去了超市"
2. 时间规范化：
   - 使用YYYY-MM-DD HH:MM:SS格式
   - 尽可能精确到秒，但原始记录中没有具体信息时也可以从后往前省略一些时间信息，或直接使用文字代替
   - 例：原句"上周三早上" → "2024-03-20 08:00:00" or "2024-03-20早上"'''
    summary: str = Field(description="""对于整个记录尽可能简短的摘要总结。

   示例：
   - 我在2024-03-15 14:30与王芳讨论了项目进度
   - 张强在2023-08-22 09:15完成了季度报告""")
    semantic_memories: list[str] = Field(description="""记录中出现的语义信息，也即知识。提取出的原子级语义单元除了需要满足上面提到的代词转换规则和时间规范化，还需满足：
1. SPO三元组结构：
   - 主语(S)：专有名词或"我"
   - 谓语(P)：动词/形容词/系动词
   - 宾语(O)：所有名词/数值/时间
2. 专有名词定义：
   - 包含人名/地名/机构名/特定事件名，"我"也算在专有名词里，因为第一人称视角需要
   - 例："我"、"上海交通大学"、"2024春季运动会"
3. 原子性要求：
   - 单句仅表达一个事实
   - 例：拆分"我今天吃了苹果和香蕉"为：
     "我吃了苹果"和"我吃了香蕉"

   示例：
   - 我毕业于北京大学
   - 李华擅长编程
   - 北京时间2024-05-01 20:00举办演唱会
   - 我的生日是1995-07-23
   - 项目截止日期为2024-06-30
   - 东京塔高度为333米""")

class ExtractedMemories(BaseModel):
    original_memories: list[MemoryEntry] = Field(description="事件原始信息")
    summary: MemoryEntry = Field(description="对于事件的概括总结。")
    semantic_memories: list[MemoryEntry] = Field(description="事件中出现的语义信息。可以有多条，或一条都没有。")


class RecycleState(BaseModel):
    #输入
    input_messages: list[AnyMessage] = Field(description="输入消息列表")
    recycle_type: Literal['extract', 'original'] = Field(description="回收类型")
    checking_messages: list[AnyMessage] = Field(default=list, description="用于检查是否溢出")
    agent_time_settings: AgentTimeSettings = Field(description="代理时间设置")
    #输出及临时状态
    remove_messages: list[RemoveMessage] = Field(default_factory=list)
    old_messages: list[AnyMessage] = Field(default_factory=list)
    new_messages: list[AnyMessage] = Field(default_factory=list)
    extracted_memories: list[MemoryEntry] = Field(default_factory=list)


class RecycleGraph(BaseGraph):

    def __init__(self, llm: BaseChatModel,
        memory_manager: MemoryManager,
        tools: Optional[Sequence[Union[Dict[str, Any], type, Callable, BaseTool]]] = None
    ):
        super().__init__(llm=llm, tools=tools, memory_manager=memory_manager)

        #self.tools = tools

        recycleGraph_builder = StateGraph(RecycleState)

        recycleGraph_builder.add_node("begin", self.begin)
        #recycleGraph_builder.add_node("recycle", self.recycle_messages)
        recycleGraph_builder.add_node("extract", self.extract_messages)
        recycleGraph_builder.add_node("original", self.original_process)
        recycleGraph_builder.add_node("extract_process", self.extract_process)
        recycleGraph_builder.add_node("store", self.store_memories)

        recycleGraph_builder.set_entry_point("begin")
        recycleGraph_builder.add_conditional_edges("begin", self.type_condition)
        #recycleGraph_builder.add_conditional_edges("original", self.original_condition)
        recycleGraph_builder.add_edge("original", "store")
        #recycleGraph_builder.add_edge("recycle", "extract")
        recycleGraph_builder.add_edge("extract_process", "extract")
        recycleGraph_builder.add_edge("extract", "store")
        recycleGraph_builder.add_edge("store", END)
        self.graph_builder = recycleGraph_builder

    @classmethod
    async def create(cls, llm: BaseChatModel, memory_manager: MemoryManager, tools: Optional[Sequence[Union[Dict[str, Any], type, Callable, BaseTool]]] = None):
        instance = cls(llm, memory_manager, tools)
        instance.conn = await aiosqlite.connect("./data/checkpoints_recycle.sqlite")
        instance.checkpointer = AsyncSqliteSaver(instance.conn)
        instance.graph = instance.graph_builder.compile(checkpointer=instance.checkpointer)
        return instance


    async def begin(self, state: RecycleState):
        return {"remove_messages": [], "old_messages": [], "new_messages": [], "extracted_memories": []}

    def type_condition(self, state: RecycleState):
        if state.input_messages:
            if state.recycle_type == 'original':
                return 'original'
            elif state.recycle_type == 'extract':
                return 'extract_process'
        return END

    # 回收溢出的消息

    async def recycle_messages(self, state: RecycleState, config: RunnableConfig):
        # 获取当前消息列表
        messages = state.checking_messages
        max_tokens = get_thread_recycle_config(config["configurable"]["thread_id"]).recycle_target_size
        new_messages = trim_messages(
            messages=messages,
            max_tokens=max_tokens,
            token_counter=count_tokens_approximately,
            strategy='last',
            start_on=HumanMessage,
            allow_partial=True,
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
        )
        if not new_messages:
            raise Exception("Trim messages failed.")
        excess_count = len(messages) - len(new_messages)
        old_messages = messages[:excess_count]
        remove_messages = [RemoveMessage(id=message.id) for message in old_messages]

        return {"remove_messages": remove_messages, "new_messages": new_messages, "old_messages": old_messages}

    async def original_process(self, state: RecycleState, config: RunnableConfig):
        messages = state.input_messages

        extracted_memories: list[MemoryEntry] = []

        current_time_seconds = now_seconds(state.agent_time_settings)
        base_stable_time = get_thread_recycle_config(config["configurable"]["thread_id"]).base_stable_time

        for message in messages:
            if isinstance(message, ToolMessage):
                if message.artifact and isinstance(message.artifact, dict):
                    if message.artifact.get("bh_do_not_store"):
                        continue
                    else:
                        stable_mult = random.uniform(0.0, 3.0)#TODO:这个值应该由文本的情感强烈程度来决定
                else:
                    stable_mult = random.uniform(0.0, 3.0)
            elif message.additional_kwargs.get("bh_do_not_store"):
                continue
            else:
                stable_mult = random.uniform(0.0, 3.0)
            bh_creation_time_seconds = message.additional_kwargs.get("bh_creation_time_seconds", current_time_seconds)
            bh_creation_agent_time_seconds = datetime_to_seconds(real_time_to_agent_time(bh_creation_time_seconds))

            extracted_memories.append(MemoryEntry(
                content=parse_message(message),
                stable_time=stable_mult * base_stable_time,
                type="original",
                creation_time_seconds=bh_creation_agent_time_seconds,
                id=message.id or str(uuid4())
            ))

        return {"extracted_memories": extracted_memories}

    def original_condition(self, state: RecycleState, config: RunnableConfig):
        if state.checking_messages:
            if count_tokens_approximately(state.checking_messages) > get_thread_recycle_config(config["configurable"]["thread_id"]).recycle_trigger_threshold:
                return 'recycle'
            else:
                return 'store'

    async def extract_process(self, state: RecycleState, config: RunnableConfig):
        return {"old_messages": state.input_messages, "remove_messages": [RemoveMessage(id=message.id) for message in state.input_messages]}

    async def extract_messages(self, state: RecycleState, config: RunnableConfig):
        messages = state.old_messages

        trimmed_messages = []

        extracted_memories = []

        current_time_seconds = now_seconds()
        base_stable_time = get_thread_recycle_config(config["configurable"]["thread_id"]).base_stable_time

        creation_time_secondses = []

        for message in messages:
            if isinstance(message, ToolMessage) and message.artifact and isinstance(message.artifact, dict) and message.artifact.get("bh_do_not_store"):
                continue
            elif message.additional_kwargs.get("bh_do_not_store"):
                continue
            elif message.additional_kwargs.get("bh_extracted"):
                continue
            trimmed_messages.append(message)
            bh_creation_time_seconds = message.additional_kwargs.get("bh_creation_time_seconds", current_time_seconds)
            bh_creation_agent_time_seconds = datetime_to_seconds(real_time_to_agent_time(bh_creation_time_seconds))
            creation_time_secondses.append(int(bh_creation_agent_time_seconds))

        if trimmed_messages:
            # 临时方案，对于总结和语义记忆的creation_time_seconds直接使用原始记录的平均时间戳
            creation_time_seconds_average = sum(creation_time_secondses) / len(creation_time_secondses)

            llm_with_structure = self.llm.with_structured_output(ExtractedMemoryInfo, method="function_calling")
            extracted_memory_info = await llm_with_structure.ainvoke(f"""以下是用户的生活记录：
<history>
{parse_messages(trimmed_messages)}
</history>
请你根据这些记录，提取出 ExtractedMemoryInfo 需要的记忆信息并返回。"""
            )

            extracted_memory_info_dict = extracted_memory_info.model_dump()

            extracted_memories.append(MemoryEntry(
                content=extracted_memory_info_dict["summary"],
                stable_time=random.uniform(0.0, 3.0) * base_stable_time,
                type="summary",
                creation_time_seconds=creation_time_seconds_average,
                id=str(uuid4())
            ))

            for semantic_memory in extracted_memory_info_dict["semantic_memories"]:
                extracted_memories.append(MemoryEntry(
                    content=semantic_memory,
                    stable_time=random.uniform(0.0, 3.0) * base_stable_time,
                    type="semantic",
                    creation_time_seconds=creation_time_seconds_average,
                    id=str(uuid4())
                ))

        return {"extracted_memories": extracted_memories}


    async def store_memories(self, state: RecycleState, config: RunnableConfig):
        memory_entries = state.extracted_memories
        if not memory_entries:
            return {"input_messages": []}
        docs = []
        for memory_entry in memory_entries:
            docs.append(self.memory_manager.InitialMemory(
                content=memory_entry.content,
                id=memory_entry.id,
                type=memory_entry.type,
                creation_time_seconds=memory_entry.creation_time_seconds,
                stable_time=memory_entry.stable_time
            ))
        await self.memory_manager.add_memories(docs, config["configurable"]["thread_id"], state.agent_time_settings)
        return {"input_messages": []}