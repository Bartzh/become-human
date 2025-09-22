from become_human.graph_base import BaseGraph
from become_human.memory import MemoryManager
from become_human.time import datetime_to_seconds, parse_seconds
from become_human.store_settings import RetrieveMemoriesConfig
from become_human.store_manager import store_manager

from typing import Any, Callable, Dict, Optional, Sequence, Union, Annotated, Literal
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool, InjectedToolArg
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    RemoveMessage
)
from langgraph.graph.state import StateGraph, START, END
from langgraph.graph.message import add_messages

#from langgraph.prebuilt import ToolNode, tools_condition, InjectedState

from datetime import datetime, timezone
import random

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

#from uuid import uuid4

class RetrieveMemoriesInput(BaseModel):
    search_string: Optional[str] = Field(default=None, description="要检索的内容")
    #increase_weight: bool = Field(default=False, description="可选，增加检索结果中该种类型记忆出现的比重")
    create_time_range: Optional[str] = Field(default=None, description='限定检索结果时间范围，格式为"2023-01-01 00:00:00~2023-01-01 23:59:59"，也即%Y-%m-%d %H:%M:%S~%Y-%m-%d %H:%M:%S')

class RetrieveMemoriesInputs(BaseModel):
    original_retrieval: Optional[RetrieveMemoriesInput] = Field(default=None, description="original类型记忆的检索输入")
    summary_retrieval: Optional[RetrieveMemoriesInput] = Field(default=None, description="summary类型记忆的检索输入")
    semantic_retrieval: Optional[RetrieveMemoriesInput] = Field(default=None, description="semantic类型记忆的检索输入")
    #state: Annotated[RetrieveState, InjectedState] = Field(description="图状态，不应展示给LLM")


class RetrieveState(BaseModel):
    #输入
    input: str = Field(description="检索输入")
    type: Literal['passive', 'active'] = Field(description="检索类型：主动或被动['passive', 'active']")
    #临时
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list, description="检索过程中产生的消息")
    output: list[Document] = Field(default_factory=list, description="检索结果")
    error: str = Field(default="", description="错误信息")

class RetrieveGraph(BaseGraph):

    def __init__(self, llm: BaseChatModel,
        memory_manager: MemoryManager,
        tools: Optional[Sequence[Union[Dict[str, Any], type, Callable, BaseTool]]] = None
    ):
        super().__init__(llm=llm, memory_manager=memory_manager, tools=tools)

        retrieveGraph_builder = StateGraph(RetrieveState)
        retrieveGraph_builder.add_node("begin", self.begin)
        retrieveGraph_builder.add_node("active_processing", self.active_processing)
        retrieveGraph_builder.add_node("passive_processing", self.passive_processing)
        retrieveGraph_builder.add_node("final", self.final)
        #tool_node = ToolNode(tools=self.tools)
        #retrieveGraph_builder.add_node("tools", tool_node)
        retrieveGraph_builder.set_entry_point("begin")
        retrieveGraph_builder.add_conditional_edges(
            "begin",
            self.route_input_type,
            {
                "active": "active_processing",
                "passive": "passive_processing"
            }
        )
        retrieveGraph_builder.add_edge("active_processing", "final")
        #retrieveGraph_builder.add_conditional_edges("tools", self.route_tools)
        retrieveGraph_builder.add_edge("passive_processing", "final")
        retrieveGraph_builder.add_edge("final", END)
        self.graph_builder = retrieveGraph_builder

    @classmethod
    async def create(cls, llm: BaseChatModel, memory_manager: MemoryManager, tools: Optional[Sequence[Union[Dict[str, Any], type, Callable, BaseTool]]] = None):
        instance = cls(llm, memory_manager, tools)
        instance.conn = await aiosqlite.connect("./data/checkpoints_retrieve.sqlite")
        instance.graph = instance.graph_builder.compile(checkpointer=AsyncSqliteSaver(instance.conn))
        return instance

    async def begin(self, state: RetrieveState):
        return {"messages": [RemoveMessage(id="__remove_all__")], "output": [], "error": ""}

    def route_input_type(self, state: RetrieveState):
        # 根据input_type字段的值路由到不同的节点
        return state.type

    async def active_processing(self, state: RetrieveState, config: RunnableConfig):
        # 主动输入的处理逻辑
        # 可以在这里添加特定于主动输入的处理代码
        thread_id = config["configurable"]["thread_id"]
        llm_with_structure = self.llm.with_structured_output(RetrieveMemoriesInputs, method="function_calling")
        retrieve_prompt = f'''你是一个高级记忆检索优化器（RetrieverPrompter），你的核心目标是负责分析用户输入并将其转换为更优的记忆检索工具的输入参数（RetrieveMemoriesInput），使得检索工具能够获得比使用原始输入直接检索更好的准确性和相关性。
**以下是输出要求：**
<rules>
# 对于输入参数的解释
1. search_string: Optional[str] = None
    用于检索记忆的查询字符串。
    准确来说是用于被向量化后在向量数据库中与其他向量做相似性算法（cosine）后返回最相似的一些结果的字符串。
    如果输入为空，则会使用create_time_range直接获取限定时间范围的一些记忆。如果create_time_range也为空，则输入无效。

2. create_time_range: Optional[str] = None
    用于限定搜索结果的时间范围，格式为"2023-01-01 00:00:00~2023-01-01 23:59:59"，也即%Y-%m-%d %H:%M:%S~%Y-%m-%d %H:%M:%S
    默认为空，表示不限制时间范围。
    由于是使用filter直接在程序上过滤掉了这个范围以外的记忆，所以比起在search_string中描述时间范围，使用create_time_range会更快且更准确。
    但也需要注意这里的create_time，准确来说指的是记忆的创建时间，这个时间是在记忆被创建时自动生成的。
    而如果出现“张三上星期感冒了。”这种情况，可能记忆是在"2025-02-12 11:31:12"时创建的，但实际上这是在记忆创建时间一个星期前发生的事件。
    这时如果使用create_time_range来限定时间范围为"2025-02-03 00:00:00~2025-02-09 23:59:59"，那么就会检索不到“张三上星期感冒了”这条记忆，因为它是在一星期后创建的。
    总之不建议在较小范围使用create_time_range，可能会出现这种意外情况，create_time_range本质是一种性能优化手段。

# 有三种类型记忆可供查询，original，summary，semantic。以下是三种不同类型记忆的说明以及它们在存储时的规范要求提示词，你可以根据这些信息来决定自己要检索哪些记忆类型并模仿创建更接近记忆内容的查询语句，因为是相似性搜索，所以使用与要检索的内容相似的语句来查询，可以获到更好的结果。
1. 对于original类型的记忆
    original类型的记忆是直接按对话轮次原文存储的，没有任何后处理也没有任何规律，所以你大概可以直接使用原始输入作为search_string。
    但这也意味着orginal类型记忆可能在大多数情况下不是特别适用，因为其一般包含大量无用信息，不如summary和semantic类型精炼。
    不过如果想回忆完整记忆，这也是唯一的方式。
    如果想回忆大致某时段的记忆，可以使用create_time_range来限定时间范围，然后将search_string留空，这样会直接获取限定时间范围内的一些记忆。

2. 对于summary类型的记忆
    summary类型的记忆是经过精炼的对于一段时间上下文的总结、摘要，也就是一段时间内发生了什么。通常包含更重要的信息，比如主题、时间、地点、人物等。
    所以summary类型记忆更适合用于查询事件大致内容。
    记忆里可能不会包含时间信息，如果需要根据时间查询，可使用create_time_range限定时间范围来获得更精确的搜索结果。

    以下是summary类型的记忆的存储规范提示词供参考：

        请以第一人称视角提取记忆信息，提取到的所有记忆信息需满足以下规则：
        1. 代词转换规则：
        - 摘要中出现的"我"只能指代当前用户（即需要提取记忆信息的主体），这是为了保持第一人称视角
        - 摘要中除"我"外，所有代词（你/他/她/他们）必须替换为完整姓名
        - 例：原句"他昨天去了超市" → "李四在2023-04-05去了超市"
        2. 时间规范化：
        - 使用YYYY-MM-DD HH:MM:SS格式
        - 尽可能精确到秒，但原始记录中没有具体信息时也可以从后往前省略一些时间信息，或直接使用文字代替
        - 例：原句"上周三早上" → "2024-03-20 08:00:00" or "2024-03-20早上"
        - 也可以两个时间中间加一个波浪号 ~ 来表示时间范围
        - 例：原句"上周三早上" → "2024-03-20 06:00:00 ~ 2024-03-20 12:00:00"

        示例：
        - 我在2024-03-15 14:30与王芳讨论了项目进度
        - 张强在2023-08-22 09:15完成了季度报告

3. 对于semantic类型的记忆
    semantic类型的记忆会比summary更进一步，是更简练的，原子性的语义信息。
    semantic类型的记忆非常适合获取知识，如果你只是想知道什么是什么，检索semantic类型记忆是最合适的。
    也因此这可能是最常用的检索类型。

    以下是semantic类型的记忆的存储规范提示词供参考：
        记录中出现的语义信息，也即知识。提取出的原子级语义单元除了需要满足summary中提到的代词转换规则和时间规范化，还需满足：
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
        - 东京塔高度为333米"

    检索示例：
    - 我喜欢什么？
    - 我毕业于哪所大学？
    - 李华擅长什么？
</rules>
**以下是用户输入：**
<user_input>
{state.input}
</user_input>'''
        max_retries = 3
        retry_count = 0
        memories = None
        before_time = datetime.now(timezone.utc)
        store_settings = await store_manager.get_settings(thread_id)
        active_retrieve_config = store_settings.retrieve.active_retrieve_config
        while retry_count < max_retries:
            try:
                retrieve_inputs = await llm_with_structure.ainvoke(retrieve_prompt)
                memories = await self.retrieve_memories(retrieve_inputs, active_retrieve_config, thread_id)
                break
            except ValueError as e:
                retry_count += 1
                retrieve_prompt = f'''{retrieve_prompt}



**[ERROR]记忆检索工具执行出错了！请参考错误信息修改你的原输出内容并重新输出参数！**（重试次数{retry_count}/{max_retries}）
错误信息：
{e}

你的原输出内容：
{retrieve_inputs.model_dump_json(indent=4)}'''

        if retry_count >= max_retries:
            after_time = datetime.now(timezone.utc)
            return {'error': f'主动检索记忆失败，这很少见。操作耗时{parse_seconds(after_time - before_time)}，请考虑是否要重新尝试。'}
        else:
            return {"output": memories}


    async def passive_processing(self, state: RetrieveState, config: RunnableConfig):
        # 被动输入的处理逻辑
        # 可以在这里添加特定于被动输入的处理代码
        thread_id = config["configurable"]["thread_id"]
        store_settings = await store_manager.get_settings(thread_id)
        retrieve_inputs = RetrieveMemoriesInputs(
            original_retrieval=RetrieveMemoriesInput(search_string=state.input),
            summary_retrieval=RetrieveMemoriesInput(search_string=state.input),
            semantic_retrieval=RetrieveMemoriesInput(search_string=state.input),
        )
        try:
            passive_retrieve_config = store_settings.retrieve.passive_retrieve_config
            memories = await self.retrieve_memories(retrieve_inputs, passive_retrieve_config, thread_id)
        except ValueError as e:
            memories = "被动检索记忆失败，请无视此消息。"
            print('被动检索报错：' + str(e))
            return {"error": memories}
        return {"output": memories}

    async def final(self, state: RetrieveState):
        return


    async def retrieve_memories(self, retrieve_inputs: RetrieveMemoriesInputs, retrieve_config: RetrieveMemoriesConfig, thread_id: str) -> list[Document]:

        total_k = retrieve_config.k
        fetch_k = retrieve_config.fetch_k
        search_method = retrieve_config.search_method
        similarity_weight = retrieve_config.similarity_weight
        retrievability_weight = retrieve_config.retrievability_weight
        diversity_weight = retrieve_config.diversity_weight
        retrieve_strength = retrieve_config.strength

        original_retrieval = retrieve_inputs.original_retrieval
        summary_retrieval = retrieve_inputs.summary_retrieval
        semantic_retrieval = retrieve_inputs.semantic_retrieval


        # 收集有效的search_string
        valid_searches = [
            r.search_string for r in [original_retrieval, summary_retrieval, semantic_retrieval] 
            if r and r.search_string.strip()
        ]
        if not valid_searches:
            raise ValueError("所有检索类型的search_string均为空或未提供，请至少提供一个有效的search_string。")

        def _process_retrieval(retrieval: Optional[RetrieveMemoriesInput], retrieval_type, default_search) -> dict:
            if retrieval:
                if retrieval.search_string.strip() or retrieval.create_time_range.strip():
                    return retrieval.model_dump() | {"type": retrieval_type, "increase_weight": True}
            else:
                return {
                    "search_string": default_search,
                    "type": retrieval_type,
                    "increase_weight": False,
                    "create_time_range": ""
                }

        default_search = valid_searches[0]

        original_retrieval_dict = _process_retrieval(original_retrieval, "original", default_search)
        summary_retrieval_dict = _process_retrieval(summary_retrieval, "summary", default_search)
        semantic_retrieval_dict = _process_retrieval(semantic_retrieval, "semantic", default_search)

        retrieves = [
            original_retrieval_dict,
            summary_retrieval_dict,
            semantic_retrieval_dict
        ]

        total_scale = 0.0

        for retrieve in retrieves:
            if retrieve["type"] == "original":
                retrieve["scale"] = retrieve_config.original_ratio
            elif retrieve["type"] == "summary":
                retrieve["scale"] = retrieve_config.summary_ratio
            elif retrieve["type"] == "semantic":
                retrieve["scale"] = retrieve_config.semantic_ratio
            else:
                retrieve["scale"] = 0.0
            if retrieve["increase_weight"]:
                retrieve["scale"] *= 1.8#TODO:这些值可以根据环境变化
            total_scale += retrieve["scale"]

        if total_scale == 0.0:
            raise ValueError("所有检索类型的权重和为0，请检查你的输入。")
        for retrieve in retrieves:
            retrieve["proportion"] = retrieve["scale"] / total_scale
            retrieve["k"] = int(total_k * retrieve["proportion"])
            retrieve["fetch_k"] = int(fetch_k * retrieve["proportion"])

        for retrieve in retrieves:
            create_time_range: str = retrieve.get("create_time_range")
            if create_time_range:
                try:
                    start_str, end_str = create_time_range.split('~')
                    retrieve["start_time"] = datetime.strptime(start_str.strip(), "%Y-%m-%d %H:%M:%S")
                    retrieve["end_time"] = datetime.strptime(end_str.strip(), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    raise ValueError(f'无法解析 {retrieve["type"]} 类型输入中的 create_time_range！请重新检查你的输入是否符合"2023-01-01 00:00:00~2023-01-01 23:59:59"这样的格式！(也即%Y-%m-%d %H:%M:%S~%Y-%m-%d %H:%M:%S)')

        for retrieve in retrieves:
            if isinstance(retrieve.get("start_time"), datetime) and isinstance(retrieve.get("end_time"), datetime):
                retrieve["filter"] = [{"stable_time": {"$gte": datetime_to_seconds(retrieve["start_time"])}}, {"stable_time": {"$lte": datetime_to_seconds(retrieve["end_time"])}}]


        docs_and_scores = await self.memory_manager.retrieve_memories(
            inputs=[
                self.memory_manager.RetrieveInput(
                    search_string=retrieve["search_string"],
                    search_type=retrieve["type"],
                    search_method=search_method,
                    k=retrieve["k"],
                    fetch_k=retrieve["fetch_k"],
                    similarity_weight=similarity_weight,
                    retrievability_weight=retrievability_weight,
                    diversity_weight=diversity_weight,
                    metadata_filter=retrieve.get("filter", []),
                    strength=retrieve_strength
                ) for retrieve in retrieves
            ],
            thread_id=thread_id
        )

        #docs = [doc for doc, _ in docs_and_scores]


        for i, doc_and_score in enumerate(docs_and_scores):
            doc = doc_and_score[0]
            score = doc_and_score[1]
            content = doc.page_content
            content_length = len(content)
            if i < retrieve_config.stable_k and score < 0.35:
                # 计算要替换的字符比例，score从0.35到0对应替换比例从0到1
                replacement_ratio = min(1 - (score / 0.35), 0.8)  # 0.35->0, 0->1
            elif i >= retrieve_config.stable_k and content_length > 15:
                length_scale = min((content_length - 15) / 35, 1.0)
                replacement_ratio = length_scale * random.uniform(0.6, 0.9)
            else:
                continue
            # 计算要替换的字符数量
            num_chars_to_replace = int(content_length * replacement_ratio)
            if num_chars_to_replace <= 0:
                continue

            # 将字符串转换为列表以便修改
            content_list = list(content)
            # 随机选择要替换的字符位置
            chars_to_replace = random.sample(range(content_length), num_chars_to_replace)
            # 将选中的字符替换为星号
            for i in chars_to_replace:
                content_list[i] = '*'
            # 重新组合成字符串
            doc.page_content = '「模糊的记忆」' + ''.join(content_list)

        docs = [doc for doc, score in docs_and_scores]

        return docs


    def route_tools(self, state: RetrieveState):
        direct_exit = True
        for message in reversed(state.messages):
            if isinstance(message, ToolMessage):
                if message.status == "error":
                    direct_exit = False
                    break
            else:
                break
        if direct_exit:
            return 'final'
        else:
            return 'active_processing'