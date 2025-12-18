from typing import Optional, Union, Any, Literal, Self
import os
import asyncio
from dateutil.relativedelta import relativedelta
from warnings import warn
from uuid import uuid4
import random

from langchain_qwq import ChatQwen, ChatQwQ
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessageChunk, HumanMessage, RemoveMessage, BaseMessage, AIMessage, AnyMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from langchain_dev_utils.chat_models import load_chat_model

from become_human.graph_main import MainGraph, send_message_tool_content, MainContext
from become_human.recycling import recycle_memories
from become_human.memory import get_activated_memory_types, memory_manager
from become_human.config import load_config, get_agent_configs
from become_human.utils import is_valid_json, format_messages_for_ai, extract_text_parts
from become_human.time import datetime_to_seconds, real_time_to_agent_time, now_seconds, format_time, utcnow, agent_seconds_to_datetime, format_seconds, Times, real_seconds_to_agent_seconds, parse_timedelta
from become_human.store import store_setup, store_stop_listener, store_adelete_namespace
from become_human.store_manager import store_manager
from become_human.tools.send_message import SEND_MESSAGE, SEND_MESSAGE_CONTENT


class AgentManager:
    """agent管理器

    有些地方会出现的thread、thread_id，跟agent是一回事。thread指langgraph checkpointer的thread，在这里就被当作agent。"""

    event_queue: asyncio.Queue

    activated_agent_ids: dict[str, dict[str, Any]]
    heartbeat_interval: float
    heartbeat_is_running: bool
    heartbeat_task: Optional[asyncio.Task]

    chat_model: BaseChatModel
    structured_model: BaseChatModel
    main_graph: MainGraph
    # 缓冲用于当双发但还没调用graph时，最后一次调用可以连上之前的输入给agent，而前面的调用直接取消即可。
    _agent_user_input_buffers: dict[str, list[str]]
    _agent_gathereds: dict[str, AIMessageChunk]
    _agent_streaming_tool_messages: dict[str, list[ToolMessage]]

    def __init__(self):
        """
        不要直接实例化此类。
        请使用 AgentManager.create() 异步创建实例。
        """
        pass

    @classmethod
    async def create(cls, heartbeat_interval: float = 5.0) -> Self:
        """
        创建AgentManager实例的唯一入口
        
        Args:
            heartbeat_interval: 心跳间隔时间
            
        Returns:
            AgentManager实例
        """
        instance = cls()
        await instance.init_manager(heartbeat_interval)
        return instance


    async def init_manager(self, heartbeat_interval: float = 5.0):

        req_envs = ["CHAT_MODEL_NAME", "STRUCTURED_MODEL_NAME"]
        for e in req_envs:
            if not os.getenv(e):
                raise Exception(f"{e} is not set")

        self.event_queue = asyncio.Queue()

        self.heartbeat_interval = heartbeat_interval
        self.activated_agent_ids = {}
        self.heartbeat_is_running = False
        self.heartbeat_task = None

        self._agent_user_input_buffers = {}
        self._agent_gathereds = {}
        self._agent_streaming_tool_messages = {}


        await store_setup()
        await load_config()

        def create_model(model_name: str, enable_thinking: Optional[bool] = None):
            splited_model_name = model_name.split(':', 1)
            if len(splited_model_name) != 2:
                raise ValueError(f"Invalid model name: {model_name}")
            else:
                provider = splited_model_name[0]
                model = splited_model_name[1]
            kwargs = {}
            if model == 'deepseek-v3.2':
                kwargs['reasoning_keep_policy'] = 'current'
            if provider == 'dashscope':
                if model.startswith(('qwen-', 'qwen3-')):
                    return ChatQwen(
                        model=model_name,
                        enable_thinking=enable_thinking,
                    )
                elif model.startswith(('qwq-', 'qvq-')):
                    return ChatQwQ(
                        model=model_name,
                    )
                else:
                    if enable_thinking:
                        kwargs['extra_body'] = {"enable_thinking": True}
                    return load_chat_model(
                        model=model,
                        model_provider='openai',
                        **kwargs,
                    )
            if model.startswith('deepseek-v3.') and enable_thinking:
                kwargs['extra_body'] = {"thinking": {"type": "enabled"}}
            else:
                return load_chat_model(
                    model=model_name,
                    **kwargs
                    #extra_body={"enable_thinking": enable_thinking} if enable_thinking is not None else None,
                )

        chat_enable_thinking = os.getenv("CHAT_MODEL_ENABLE_THINKING", '').lower()
        if chat_enable_thinking == "true":
            chat_enable_thinking = True
        elif chat_enable_thinking == "false":
            chat_enable_thinking = False
        else:
            chat_enable_thinking = None
        self.chat_model = create_model(os.getenv("CHAT_MODEL_NAME", ""), chat_enable_thinking)

        structured_enable_thinking = os.getenv("STRUCTURED_MODEL_ENABLE_THINKING", '').lower()
        if structured_enable_thinking == "true":
            structured_enable_thinking = True
        elif structured_enable_thinking == "false":
            structured_enable_thinking = False
        else:
            structured_enable_thinking = None
        self.structured_model = create_model(os.getenv("STRUCTURED_MODEL_NAME", ""), structured_enable_thinking)

        self.main_graph = await MainGraph.create(self.chat_model, llm_for_structured_output=self.structured_model)

        # 启动heartbeat
        for key, value in get_agent_configs().items():
            if value.get('init_on_startup'):
                await self.init_agent(key)
        if self.heartbeat_task is None:
            self.heartbeat_task = asyncio.create_task(self.start_heartbeat_task())


    async def start_heartbeat_task(self):
        if self.heartbeat_is_running and self.heartbeat_task is not None:
            return
        self.heartbeat_is_running = True
        while self.heartbeat_is_running:
            await self.trigger_agents()
            await asyncio.sleep(self.heartbeat_interval)
        print("Heartbeat task stopped.")


    async def trigger_agents(self):
        tasks = [self.trigger_agent(agent_id) for agent_id, value in self.activated_agent_ids.items() if value.get("initialized")]
        await asyncio.gather(*tasks)

    async def trigger_agent(self, agent_id: str):
        config = {"configurable": {"thread_id": agent_id}}
        agent_store = await store_manager.get_agent(agent_id)
        agent_settings = agent_store.settings
        agent_states = agent_store.states

        # 获取时间
        current_datetime = utcnow()
        current_time_seconds = datetime_to_seconds(current_datetime)
        current_agent_datetime = real_time_to_agent_time(current_datetime, agent_settings.main.time_settings)
        current_agent_time_seconds = datetime_to_seconds(current_agent_datetime)

        # 如果是首次运行，则添加或发送引导消息
        if agent_states.is_first_time:
            agent_states.is_first_time = False
            instruction_message = HumanMessage(
                content=f'''**这条消息来自系统（system）自动发送**
当前时间是：{format_time(current_agent_datetime)}。
这是你被初始化以来的第一条消息。如果你看到这条消息，说明在此消息之前你还没有收到过任何来自用户的消息。
这意味着你的“记忆”暂时是空白的，如果检索记忆时提示“没有找到任何匹配的记忆。”或检索不到什么有用的信息，这是正常的。
接下来是你与用户的初次见面，请根据你所扮演的角色以及以下的提示考虑应做出什么反应：\n''' + agent_settings.main.instruction_prompt,
                additional_kwargs={
                    "bh_creation_time_seconds": current_time_seconds,
                    "bh_creation_agent_time_seconds": current_agent_time_seconds,
                    "bh_from_system": True,
                    "bh_do_not_store": True
                },
                name='system'
            )
            if agent_settings.main.react_instruction:
                await self.call_agent(instruction_message.text, agent_id, call_type='system')
            else:
                await self.main_graph.update_messages(agent_id, [instruction_message])

        # 更新记忆
        await self.process_timers(agent_id)

        # 处理每天的任务
        last_update_agent_datetime = agent_seconds_to_datetime(agent_states.last_update_agent_time_seconds, agent_settings.main.time_settings)
        if (current_agent_datetime.day != last_update_agent_datetime.day or
            current_agent_datetime.month != last_update_agent_datetime.month or
            current_agent_datetime.year != last_update_agent_datetime.year):

            # 处理年龄（我觉得年龄应该靠自己想，而非程序计算）
            if agent_settings.main.character_settings.birthday is not None:
                age = relativedelta(current_agent_datetime.date(), agent_settings.main.character_settings.birthday).years
                if age != agent_settings.main.character_settings.age:
                    agent_settings.main.character_settings.age = age

        # 更新最后更新时间
        agent_states.last_update_real_time_seconds = current_time_seconds
        agent_states.last_update_agent_time_seconds = current_agent_time_seconds

        # 如果agent的main_graph正在运行，取消以下任务
        main_graph_state = await self.main_graph.graph.aget_state(config)
        if main_graph_state.next:
            return

        # 处理自我调用
        self_call_time_secondses = main_graph_state.values.get("self_call_time_secondses", [])
        wakeup_call_time_seconds = main_graph_state.values.get("wakeup_call_time_seconds")
        active_self_call_time_secondses_and_notes = main_graph_state.values.get("active_self_call_time_secondses_and_notes", [])
        can_call = False
        if active_self_call_time_secondses_and_notes:
            for seconds, note in active_self_call_time_secondses_and_notes:
                if current_agent_time_seconds >= seconds:
                    can_call = True
                    self_call_type = 'active'
                    break
        if self_call_time_secondses and not can_call:
            for seconds in self_call_time_secondses:
                if current_agent_time_seconds >= seconds:
                    can_call = True
                    self_call_type = 'passive'
                    break
        if wakeup_call_time_seconds and not can_call:
            if current_agent_time_seconds >= wakeup_call_time_seconds:
                can_call = True
                self_call_type = 'passive'
        if can_call:
            await self.call_agent('', agent_id, call_type='self', self_call_type=self_call_type)

        # 自动回收闲置上下文
        elif main_graph_state.values.get("active_time_seconds") and current_agent_time_seconds > main_graph_state.values.get("active_time_seconds"):# 已超出活跃时间
            messages = await self.main_graph.get_messages(agent_id)
            # 最后一条消息如果为HumanMessage说明agent还没有响应
            if not isinstance(messages[-1], HumanMessage):
                not_extracted_messages = [m for m in messages if not m.additional_kwargs.get("bh_extracted")]
                # 再次判断是否有来自用户的新消息
                if [m for m in not_extracted_messages if isinstance(m, HumanMessage) and not m.additional_kwargs.get("bh_from_system") and not m.additional_kwargs.get("bh_do_not_store")]:

                    # 清理掉passive_retrieval消息，可以考虑加个开关
                    passive_retrieve_messages = [m for m in messages if m.additional_kwargs.get("bh_message_type", '') == "passive_retrieval"]
                    remove_messages = [RemoveMessage(id=m.id) for m in passive_retrieve_messages]

                    # recycling
                    memory_types = get_activated_memory_types()
                    recycles = {t: recycle_memories(t, agent_id, not_extracted_messages, self.structured_model) for t in memory_types}
                    recycle_results = {}
                    if len(recycles) > 0:
                        graph_results = await asyncio.gather(*recycles.values())
                        recycle_results = {k: graph_results[i] for i, k in enumerate(recycles.keys())}

                    # cleanup
                    is_cleanup = agent_settings.recycling.cleanup_on_non_active_recycling
                    if is_cleanup:
                        #await main_graph.update_messages(agent_id, [RemoveMessage(id=m.id) for m in messages])
                        max_tokens = agent_settings.recycling.cleanup_target_size
                        if max_tokens > 0:
                            # 因为已经决定清理passive_retrieval消息了，所以这里直接以清理之后的messages计算
                            not_passive_retrieve_messages = [m for m in messages if m.additional_kwargs.get("bh_message_type", '') != "passive_retrieval"]
                            if count_tokens_approximately(not_passive_retrieve_messages) > max_tokens:
                                new_messages: list[BaseMessage] = trim_messages(
                                    messages=not_passive_retrieve_messages,
                                    max_tokens=max_tokens,
                                    token_counter=count_tokens_approximately,
                                    strategy='last',
                                    start_on=HumanMessage,
                                    #allow_partial=True,
                                    #text_splitter=RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
                                )
                                if not new_messages:
                                    warn("Trim messages failed on cleanup.")
                                    new_messages = []
                                excess_count = len(not_passive_retrieve_messages) - len(new_messages)
                                old_messages = not_passive_retrieve_messages[:excess_count]
                                remove_messages.extend([RemoveMessage(id=message.id) for message in old_messages])
                                #update_messages = new_messages
                        # max_tokens <= 0 则全部删除
                        else:
                            remove_messages = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]

                    # 更新与清理
                    for m in not_extracted_messages:
                        m.additional_kwargs["bh_extracted"] = True
                    await self.main_graph.update_messages(agent_id, not_extracted_messages + remove_messages)

                    # 若有，将reflective的思考过程加入messages
                    if recycle_results.get('reflective'):
                        await self.main_graph.update_messages(agent_id, recycle_results.get('reflective', []))

        # 闲置过久（两个星期）则关闭agent
        elif (
            agent_id in self.activated_agent_ids and
            current_time_seconds > (self.activated_agent_ids.get(agent_id, {}).get("created_at", 0) + 1209600)
        ):
            self.close_agent(agent_id)


    async def init_agent(self, agent_id: str):
        self.activated_agent_ids[agent_id] = {"initialized": False, "created_at": now_seconds()}
        await store_manager.init_agent(agent_id)
        await self.trigger_agent(agent_id)
        if agent_id in self.activated_agent_ids:
            self.activated_agent_ids[agent_id]["initialized"] = True

    def close_agent(self, agent_id: str):
        """手动关闭agent"""
        if self.activated_agent_ids.get(agent_id):
            del self.activated_agent_ids[agent_id]
        store_manager.close_agent(agent_id)

    async def close_manager(self):
        print("wait for the last heartbeat to stop")
        self.heartbeat_is_running = False
        if self.heartbeat_task is not None:
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None

        await self.main_graph.conn.close()
        await store_stop_listener()
        print("Graphs closed")

    async def close_manager_force(self):
        self.heartbeat_is_running = False
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None

        await self.main_graph.conn.close()
        await store_stop_listener()
        print("Graphs closed")


    async def call_agent_with_command(
            self,
            user_input: Union[str, list[str]],
            agent_id: str,
            is_admin: bool = False,
            user_name: Optional[str] = None,
            call_type: Literal['human', 'self', 'system'] = 'human',
            self_call_type: Literal['passive', 'active'] = 'passive'
        ):
        extracted_message = extract_text_parts(user_input)
        if extracted_message and extracted_message[0].startswith("/"):
            if is_admin:
                await self.command_processing(agent_id, extracted_message[0])
            else:
                await self.event_queue.put({"agent_id": agent_id, "name": "log", "args": {"content": "无权限执行此命令"}, "id": "command-" + str(uuid4())})
        else:
            await self.call_agent(
                user_input=user_input,
                agent_id=agent_id,
                call_type=call_type,
                user_name=user_name,
                self_call_type=self_call_type
            )

    async def call_agent(
            self,
            user_input: Union[str, list[str]],
            agent_id: str,
            user_name: Optional[str] = None,
            call_type: Literal['human', 'self', 'system'] = 'human',
            self_call_type: Literal['passive', 'active', 'wakeup'] = 'passive'
        ):

        if call_type == 'human':
            # 首先把user_input加入buffer
            if not self._agent_user_input_buffers.get(agent_id):
                self._agent_user_input_buffers[agent_id] = []
            if isinstance(user_input, list):
                self._agent_user_input_buffers[agent_id].extend(user_input)
            else:
                self._agent_user_input_buffers[agent_id].append(user_input)
        elif self.main_graph.agent_run_ids.get(agent_id) and self_call_type != 'active':
            # 如果是自我调用同时已有运行，则不重复运行。主动自我调用则不处理
            return

        # 生成运行id并加入运行id字典
        agent_run_id = str(uuid4())
        self.main_graph.agent_run_ids[agent_id] = agent_run_id

        # 当前agent有gathered，说明是在chatbot节点处打断了上次运行，则将上次运行结果加入中断数据字典
        if self._agent_gathereds.get(agent_id):
            interrupt_data = {}
            interrupt_data['chunk'] = self._agent_gathereds.pop(agent_id)
            if self._agent_streaming_tool_messages.get(agent_id):
                interrupt_data['called_tool_messages'] = self._agent_streaming_tool_messages.pop(agent_id)
            if interrupt_data:
                self.main_graph.agent_interrupt_datas[agent_id] = interrupt_data
        # 如果有未处理的工具消息，则清理。这是意外情况
        elif self._agent_streaming_tool_messages.get(agent_id):
            del self._agent_streaming_tool_messages[agent_id]
            warn('agent已结束但有未处理的工具消息，已清理')

        # 随机时长的等待，模拟人不会一直盯着新消息，也防止短时间的双发
        if call_type == 'human':
            await asyncio.sleep(random.uniform(1.0, 4.0))
            # 如果在等待期间又有新的调用，则取消这次调用
            if self.main_graph.agent_run_ids.get(agent_id, '') != agent_run_id:
                return

        # 如果main_graph正在运行"tools"节点，则等待其运行完毕再打断。
        config = {"configurable": {"thread_id": agent_id}}
        main_graph_state = await self.main_graph.graph.aget_state(config)
        if main_graph_state.next:
            current_node = main_graph_state.next[0]
            while (
                current_node == "tools" or
                current_node == "tool_node_post_process" or
                current_node == "prepare_to_recycle" or
                current_node == "begin"
            ):
                await asyncio.sleep(0.2)
                # 如果在等待期间又有新的调用，则取消这次调用
                if self.main_graph.agent_run_ids.get(agent_id, '') != agent_run_id:
                    return
                main_graph_state = await self.main_graph.graph.aget_state(config)
                current_node = main_graph_state.next[0]

        # 从buffer中取出user_input
        if call_type == 'human':
            user_input = self._agent_user_input_buffers.pop(agent_id)
        else:
            if self._agent_user_input_buffers.get(agent_id):
                del self._agent_user_input_buffers[agent_id]
                warn("在自我或系统调用时意外发现存在残留未处理的用户输入，已删除。")

        # 初始化变量
        context = MainContext(agent_id=agent_id, agent_run_id=agent_run_id)
        store_settings = await store_manager.get_settings(agent_id)
        time_settings = store_settings.main.time_settings
        current_times = Times(time_settings)
        parsed_agent_time = format_time(current_times.agent_time)

        # 如果是用户调用
        if call_type == 'human':
            context.call_type = 'human'
            # 处理用户姓名
            if isinstance(user_name, str):
                user_name = user_name.strip()
            name = user_name or "未知姓名"
            # 加上时间信息
            if isinstance(user_input, str):
                input_content = f'[{parsed_agent_time}]\n{name}: {user_input}'
            elif isinstance(user_input, list):
                if len(user_input) == 1:
                    input_content = f'[{parsed_agent_time}]\n{name}: {user_input[0]}'
                elif len(user_input) > 1:
                    input_content = [{'type': 'text', 'text': f'[{parsed_agent_time}]\n{name}: {message}'} for message in user_input]
                else:
                    del self.main_graph.agent_run_ids[agent_id]
                    raise ValueError("Input list cannot be empty")
            else:
                del self.main_graph.agent_run_ids[agent_id]
                raise ValueError("Invalid input type")
            graph_input = {"input_messages": HumanMessage(
                content=input_content,
                name=user_name,
                additional_kwargs={
                    "bh_creation_time_seconds": current_times.real_time_seconds,
                    "bh_creation_agent_time_seconds": current_times.agent_time_seconds
                }
            )}
        elif call_type == 'self':
            context.call_type = 'self'
            context.self_call_type = self_call_type
            graph_input = {}
        elif call_type == 'system':
            context.call_type = 'system'
            graph_input = {"input_messages": HumanMessage(
                content=user_input,
                additional_kwargs={
                    "bh_creation_time_seconds": current_times.real_time_seconds,
                    "bh_creation_agent_time_seconds": current_times.agent_time_seconds,
                    "bh_from_system": True,
                    "bh_do_not_store": True
                },
                name='system'
            )}
        else:
            del self.main_graph.agent_run_ids[agent_id]
            raise ValueError("Invalid call type")

        first = True
        tool_index = 0
        last_message = ''
        canceled = False
        gathered = None
        streaming_tool_messages = []
        async for typ, msg in self.main_graph.graph.astream(graph_input, config=config, context=context, stream_mode=["updates", "messages"]):
            if typ == "updates":
                #print(msg)

                if not self.main_graph.agent_run_ids.get(agent_id) or self.main_graph.agent_run_ids.get(agent_id) != agent_run_id:
                    canceled = True

                if not canceled and msg.get("chatbot"):
                    first = True
                    tool_index = 0
                    gathered = None
                    streaming_tool_messages = []
                    last_message = ''
                    del self._agent_gathereds[agent_id]
                    del self._agent_streaming_tool_messages[agent_id]

                if msg.get("chatbot"):
                    messages = msg.get("chatbot").get("messages", [])
                elif msg.get("tool_node_post_process"):
                    messages = msg.get("tool_node_post_process").get("messages", [])
                else:
                    continue
                if isinstance(messages, BaseMessage):
                    messages = [messages]
                if messages:
                    for message in messages:
                        message.pretty_print()
                        if isinstance(message, AIMessage) and message.additional_kwargs.get("reasoning_content"):
                            print("reasoning_content: " + message.additional_kwargs.get("reasoning_content", ""))



            elif typ == "messages":
                #print(msg[0], end="\n\n", flush=True)

                if canceled:
                    continue

                if not self.main_graph.agent_run_ids.get(agent_id) or self.main_graph.agent_run_ids.get(agent_id) != agent_run_id:
                    canceled = True
                    continue

                if isinstance(msg[0], AIMessageChunk):

                    chunk: AIMessageChunk = msg[0]
                    now_times = Times(time_settings)
                    # 在streaming时就实时更新时间
                    # 但langchain在合并chunk中的additional_kwargs时不支持float，且对int和str都是加算
                    # 考虑不要依赖在这里添加时间
                    if (
                        (last_real_seconds := chunk.additional_kwargs.get("bh_creation_time_seconds")) and
                        (last_agent_seconds := chunk.additional_kwargs.get("bh_creation_agent_time_seconds"))
                    ):
                        chunk.additional_kwargs['bh_creation_time_seconds'] += int(now_times.real_time_seconds - last_real_seconds)
                        chunk.additional_kwargs['bh_creation_agent_time_seconds'] += int(now_times.agent_time_seconds - last_agent_seconds)
                    else:
                        chunk.additional_kwargs['bh_creation_time_seconds'] = int(now_times.real_time_seconds)
                        chunk.additional_kwargs['bh_creation_agent_time_seconds'] = int(now_times.agent_time_seconds)

                    if first:
                        first = False
                        gathered = chunk
                        self._agent_gathereds[agent_id] = chunk
                        streaming_tool_messages = []
                        self._agent_streaming_tool_messages[agent_id] = []
                    else:
                        if chunk.id != gathered.id:
                            gathered = chunk
                            streaming_tool_messages= []
                            self._agent_streaming_tool_messages[agent_id] = []
                        else:
                            gathered += chunk
                        self._agent_gathereds[agent_id] = gathered

                    tool_call_chunks = gathered.tool_call_chunks
                    tool_calls = gathered.tool_calls


                    #if chunk.response_metadata.get('finish_reason'):
                    #    continue


                    loop_once = True

                    while loop_once:

                        loop_once = False
                        if 0 <= tool_index < len(tool_calls):

                            chunk_completed = is_valid_json(tool_call_chunks[tool_index]['args'])
                            tool_call_id = tool_calls[tool_index].get('id', 'run-' + str(tool_index))

                            if tool_calls[tool_index]['name'] == SEND_MESSAGE:

                                new_message = tool_calls[tool_index]['args'].get(SEND_MESSAGE_CONTENT)

                                if new_message:
                                    # 对于event来说名字固定为send_message
                                    #await self.event_queue.put({"agent_id": agent_id, "name": "send_message", "args": {"content": new_message.replace(last_message, '', 1)}, "not_completed": True})
                                    event_item = {"agent_id": agent_id, "name": "send_message", "args": {"content": new_message}, "id": tool_call_id}
                                    # 暂时用来给app的通知服务使用，如果是自我调用就推送通知
                                    event_item["is_self_call"] = call_type == 'self'
                                    if not chunk_completed:
                                        event_item["not_completed"] = True
                                    else:
                                        if tool_calls[tool_index].get('id'):
                                            now_times = Times(time_settings)
                                            streaming_tool_messages.append(ToolMessage(
                                                content=send_message_tool_content(new_message),
                                                name=SEND_MESSAGE,
                                                tool_call_id=tool_calls[tool_index]['id'],
                                                additional_kwargs={
                                                    "bh_creation_time_seconds": now_times.real_time_seconds,
                                                    "bh_creation_agent_time_seconds": now_times.agent_time_seconds
                                                }
                                            ))
                                            self._agent_streaming_tool_messages[agent_id] = streaming_tool_messages
                                        last_message = ''
                                    await self.event_queue.put(event_item)
                                    #print(new_message.replace(last_message, '', 1), end="", flush=True)
                                    last_message = new_message


                            if chunk_completed:
                                #if tool_calls[tool_index]['name'] == SEND_MESSAGE:
                                #    last_message = ''
                                #    await self.event_queue.put({"agent_id": agent_id, "name": "send_message", "args": {"content": ""}})
                                    #print('', flush=True)
                                if hasattr(self.main_graph.streaming_tools, tool_calls[tool_index]['name']):
                                    method = getattr(self.main_graph.streaming_tools, tool_calls[tool_index]['name'])
                                    result = await method(tool_calls[tool_index]['args'])
                                    if tool_calls[tool_index].get('id'):
                                        now_times = Times(time_settings)
                                        streaming_tool_messages.append(ToolMessage(
                                            content=result,
                                            name=tool_calls[tool_index]['name'],
                                            tool_call_id=tool_calls[tool_index]['id'],
                                            additional_kwargs={
                                                "bh_creation_time_seconds": now_times.real_time_seconds,
                                                "bh_creation_agent_time_seconds": now_times.agent_time_seconds
                                            }
                                        ))
                                        self._agent_streaming_tool_messages[agent_id] = streaming_tool_messages
                                    await self.event_queue.put({"agent_id": agent_id, "name": tool_calls[tool_index]['name'], "args": tool_calls[tool_index]['args'], "id": tool_call_id})
                                    #print(await method(tool_calls[tool_index]['args']), flush=True)
                                tool_index += 1
                                loop_once = True


        if self.main_graph.agent_run_ids.get(agent_id) and self.main_graph.agent_run_ids.get(agent_id) == agent_run_id:
            del self.main_graph.agent_run_ids[agent_id]
        return

    async def process_timers(self, agent_id: str):
        await memory_manager.update_schedules(agent_id)


    async def command_processing(self, agent_id: str, user_input: str):
        config = {"configurable": {"thread_id": agent_id}}
        message = '无效命令'
        if user_input == "/help":
            message = """可用指令列表：
/help - 显示此帮助信息
/get_state <key> - 获取指定状态键值
/delete_last_messages <数量> - 删除最后几条消息
/set_role_prompt <提示词> - 设置角色提示词
/load_config [agentID|__all__] - 加载配置
/wakeup - 唤醒agent（重置自我调用状态）
/messages - 查看所有消息
/tokens - 计算消息令牌数
/memories <类型> [偏移] [数量] - 查看记忆
/reset <all|config> - 重置agent数据
/skip_agent_time <时间> - 跳过agent时间

使用 /<指令> help 查看具体使用说明"""
        elif user_input.startswith("/get_state ") or user_input == "/get_state":
            if user_input == "/get_state help":
                message = """使用方法：/get_state <key>
key: 要获取的状态键名，例如 /get_state active_time_seconds"""
            elif user_input != "/get_state":
                splited_input = user_input.split(" ")
                requested_key = splited_input[1]

                state = await self.main_graph.graph.aget_state(config)
                message = f"状态[{requested_key}]: {state.values.get(requested_key, '未找到该键')}"

        elif user_input.startswith("/delete_last_messages ") or user_input == "/delete_last_messages":
            if user_input == "/delete_last_messages help":
                message = """使用方法：/delete_last_messages <数量>
数量: 要删除的最后消息条数，例如 /delete_last_messages 3"""
            elif user_input != "/delete_last_messages":
                splited_input = user_input.split(" ")
                message_count = int(splited_input[1])
                if message_count > 0:
                    _main_messages = await self.main_graph.get_messages(agent_id)
                    if _main_messages:
                        _last_messages = _main_messages[-message_count:]
                        remove_messages = [RemoveMessage(id=_message.id) for _message in _last_messages if _message.id]
                        await self.main_graph.update_messages(agent_id, remove_messages)
                        message = f"已删除最后{len(remove_messages)}条消息。"
                    else:
                        message = "没有找到任何消息"

        elif user_input.startswith("/set_role_prompt ") or user_input == "/set_role_prompt":
            if user_input == "/set_role_prompt help":
                message = """使用方法：/set_role_prompt <提示词>
提示词: 要设置的角色提示词，例如 /set_role_prompt 你是一个友好的助手"""
            elif user_input != "/set_role_prompt":
                splited_input = user_input.split(" ", 1)
                role_prompt = splited_input[1].strip()
                if role_prompt:
                    agent_settings = await store_manager.get_settings(agent_id)
                    agent_settings.main.role_prompt = role_prompt
                    message = "角色提示词设置成功"
                else:
                    message = "角色提示词不能为空"

        elif user_input == "/load_config" or user_input.startswith("/load_config "):
            if user_input == "/load_config help":
                message = """使用方法：/load_config [agentID|__all__]
agentID: 要加载的特定agent配置
__all__: 加载所有agent配置
例如：/load_config agent_1 或 /load_config __all__"""
            else:
                if user_input == "/load_config":
                    result = await load_config(agent_id, force=True)
                else:
                    splited_input = user_input.split(" ")
                    if splited_input[1]:
                        if splited_input[1] == "__all__":
                            result = await load_config(force=True)
                        else:
                            result = await load_config(splited_input[1], force=True)
                if result:
                    await store_manager.init_agent(agent_id)
                    message = "配置文件已加载。"
                else:
                    message = "不存在指定的agentID。"

        elif user_input == "/wakeup" or user_input == "/wakeup help":
            if user_input == "/wakeup help":
                message = """使用方法：/wakeup
唤醒agent（重置agent的自我调用状态），这可能导致agent对prompt有些误解。"""
            else:
                await self.main_graph.graph.aupdate_state(
                    config,
                    {
                        "active_time_seconds": 0.0,
                        "self_call_time_secondses": [],
                        "wakeup_call_time_seconds": 0.0
                    },
                    as_node='final'
                )
                message = "已唤醒agent（重置自我调用相关状态），这可能导致agent对prompt有些误解。"

        elif user_input == "/messages" or user_input == "/messages help":
            if user_input == "/messages help":
                message = """使用方法：/messages
查看agent消息列表中的所有消息。"""
            else:
                main_messages = await self.main_graph.get_messages(agent_id)
                if main_messages:
                    time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
                    message = format_messages_for_ai(main_messages, time_settings)
                else:
                    message = "agent消息列表为空。"

        elif user_input == "/tokens" or user_input == "/tokens help":
            if user_input == "/tokens help":
                message = """使用方法：/tokens
计算agent消息列表的token数量。"""
            else:
                main_messages = await self.main_graph.get_messages(agent_id)
                if main_messages:
                    message = str(count_tokens_approximately(main_messages))
                else:
                    message = "agent消息列表为空，无法计算token数量。"

        elif user_input.startswith("/memories ") or user_input == "/memories":
            if user_input == "/memories help":
                message = """使用方法：/memories <类型> [偏移] [数量]
类型: original(原始记忆), summary(记忆摘要), semantic(语义记忆)
偏移: 可选，从第几条开始获取，默认0
数量: 可选，获取多少条，默认6
例如：/memories original 0 3"""
            elif user_input != "/memories":
                splited_input = user_input.split(" ")
                memory_type = splited_input[1]
                limit = int(splited_input[3]) if len(splited_input) > 3 else 6
                offset = int(splited_input[2]) if len(splited_input) > 2 else None
                get_result = await memory_manager.aget(agent_id=agent_id, memory_type=memory_type, limit=limit, offset=offset)
                message = '\n\n\n'.join([f'''id: {get_result["ids"][i]}

content: {get_result["documents"][i]}

stable_time: {get_result["metadatas"][i]["stable_time"]}

retrievability: {get_result["metadatas"][i]["retrievability"]}''' for i in range(len(get_result["ids"]))])
                if not message:
                    message = "没有找到任何记忆。"

        elif user_input == "/reset" or user_input.startswith("/reset "):
            if user_input == "/reset help":
                message = """使用方法：/reset <all|config>
all: 重置该agent所有数据（运行时不可用）
config: 仅重置配置
例如：/reset config 或 /reset all"""
            elif user_input != "/reset":
                splited_input = user_input.split(" ")
                if len(splited_input) >= 2 and splited_input[1]:
                    reset_type = splited_input[1]
                    if reset_type == 'config':
                        await store_adelete_namespace((agent_id, 'model', 'settings'))
                        await store_manager.init_agent(agent_id)
                        message = "已重置该agent配置。"
                    elif reset_type == 'all':
                        main_state = await self.main_graph.graph.aget_state(config)
                        if not main_state.next:
                            self.close_agent(agent_id)
                            await self.main_graph.graph.checkpointer.adelete_thread(agent_id)
                            memory_manager.delete_collection(agent_id, "original")
                            memory_manager.delete_collection(agent_id, "summary")
                            memory_manager.delete_collection(agent_id, "semantic")
                            await store_adelete_namespace(('agents', agent_id))
                            await load_config(agent_id)
                            await self.init_agent(agent_id)
                            message = "已重置该agent所有数据。"
                        else:
                            message = "agent运行时无法重置所有数据。"

        elif user_input == "/skip_agent_time" or user_input.startswith("/skip_agent_time "):
            if user_input == "/skip_agent_time help":
                message = """使用方法：/skip_agent_time <时间>
时间: 要跳过的时间，格式为`1w2d3h4m5s`，意为1周2天3小时4分钟5秒（也可有小数）。例如 /skip_agent_time 1w1.5d，意为跳过1周加1.5天。
注意：在涉及现实时间的一些场景如网络搜索时agent可能会感到混乱"""
            elif user_input != "/skip_agent_time":
                splited_input = user_input.split(" ", 1)
                delta_str = splited_input[1]
                try:
                    parsed_time_seconds = parse_timedelta(delta_str).total_seconds()
                except ValueError:
                    message = "时间格式错误，请确认格式正确，如 1w2d3h4m5s。"
                store_settings = await store_manager.get_settings(agent_id)
                time_settings = store_settings.main.time_settings
                current_time_seconds = now_seconds()
                new_agent_time_anchor = real_seconds_to_agent_seconds(current_time_seconds, time_settings) + parsed_time_seconds
                store_settings.main.time_settings = time_settings.model_copy(
                    update={
                        "agent_time_anchor": new_agent_time_anchor,
                        "real_time_anchor": current_time_seconds
                    },
                    deep=True
                )
                message = f"已使agent时间跳过了{format_seconds}。"


        #return {"name": "log", "args": {"message": message}}
        await self.event_queue.put({"agent_id": agent_id, "name": "log", "args": {"content": message}, "id": 'command-' + str(uuid4())})
