from typing import Optional, Union, Any, Literal
import os
import asyncio
from dateutil.relativedelta import relativedelta
from loguru import logger
from uuid import uuid4
import random

from langchain_qwq import ChatQwen, ChatQwQ
from langchain_core.messages import AIMessageChunk, HumanMessage, RemoveMessage, BaseMessage, AIMessage, AnyMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from langchain_dev_utils.chat_models import load_chat_model

from become_human.types.main import MainContext, InterruptData, MainState
from become_human.types.agent_manager import CallAgentKwargs
from become_human.graphs.base import StateMerger
from become_human.graphs.main import MainGraph, SEND_MESSAGE_TOOL_CONTENT
from become_human.recycling import recycle_memories
from become_human.memory import get_activated_memory_types, memory_manager, construct_default_memory_update_schedules
from become_human.config import load_config, get_init_on_startup_agent_ids, get_agent_enabled_plugin_names
from become_human.utils import is_valid_json
from become_human.times import format_time, format_duration, Times, parse_timedelta, AgentTimeSettings, timedelta_to_microseconds, TimestampUs
from become_human.message import format_messages_for_ai, extract_text_parts, construct_system_message, BH_MESSAGE_METADATA_KEY, BHMessageMetadata
from become_human.store.base import store_setup, store_stop_listener, store_adelete_namespace
from become_human.store.manager import store_manager
from become_human.tools.send_message import SEND_MESSAGE, SEND_MESSAGE_CONTENT
from become_human.scheduler import get_schedules, tick_schedules, delete_schedules, add_schedules, init_schedules_db, Schedule, update_schedules
from become_human.plugin import Plugin, Cancelled


class AgentManager:
    """agent管理器

    有些地方会出现的thread、thread_id，跟agent是一回事。thread指langgraph checkpointer的thread，在这里就被当作agent。"""

    event_queue: asyncio.Queue
    plugins_with_name: dict[str, Plugin]

    activated_agent_id_datas: dict[str, dict[str, Any]]
    heartbeat_interval: float
    heartbeat_is_running: bool
    heartbeat_task: Optional[asyncio.Task]

    # 两者目前来说是一样的
    on_heartbeat_finished: asyncio.Event
    on_trigger_agents_finished: asyncio.Event

    chat_model: BaseChatModel
    structured_model: BaseChatModel
    main_graph: MainGraph
    main_graph_state_merger: StateMerger
    # 缓冲用于当双发但还没调用graph时，最后一次调用可以连上之前的输入给agent，而前面的调用直接取消即可。
    _call_agent_buffers: dict[str, list[CallAgentKwargs]]
    _agent_interrupt_datas: dict[str, InterruptData]
    _agent_run_id_on_before_call_agent: dict[str, str]

    def __init__(self):
        """
        不要直接实例化此类。
        请导入 agent_manager 实例变量，再调用实例方法 init_manager 完成初始化。
        """
        raise NotImplementedError("""不要直接实例化此类！
请导入 agent_manager 实例变量，再调用实例方法 init_manager 完成初始化。""")


    async def init_manager(self, plugins: Optional[list[Union[type[Plugin], Plugin]]] = None, heartbeat_interval: float = 5.0):
        logger.info("Initializing agent manager...")

        req_envs = ["CHAT_MODEL_NAME", "STRUCTURED_MODEL_NAME"]
        for e in req_envs:
            if not os.getenv(e):
                raise Exception(f"{e} is not set")

        self.event_queue = asyncio.Queue()

        self.heartbeat_interval = heartbeat_interval
        self.activated_agent_id_datas = {}
        self.heartbeat_is_running = False
        self.heartbeat_task = None

        self.on_heartbeat_finished = asyncio.Event()
        self.on_heartbeat_finished.set()
        self.on_trigger_agents_finished = asyncio.Event()
        self.on_trigger_agents_finished.set()

        self._call_agent_buffers = {}
        self._agent_interrupt_datas = {}
        self._agent_run_id_on_before_call_agent = {}


        await store_setup()
        await init_schedules_db()

        if plugins is None:
            plugins = []
        plugins_with_name = {}
        for plugin in plugins:
            # name必须是类属性
            if plugin.name not in plugins_with_name:
                plugins_with_name[plugin.name] = plugin() if isinstance(plugin, type) else plugin
            else:
                raise ValueError(f"Plugin name {plugin.name} is duplicated.")
        self.plugins_with_name = plugins_with_name

        await load_config(self.plugins_with_name)

        store_namespaces = set(['builtin'])
        for name, plugin in self.plugins_with_name.items():
            if hasattr(plugin, 'config'):
                if plugin.config._namespace not in store_namespaces:
                    store_namespaces.add(plugin.config._namespace)
                    await store_manager.register_model(plugin.config)
                else:
                    raise ValueError(f"Plugin {name} config namespace {plugin.config._namespace} is duplicated.")
            if hasattr(plugin, 'data'):
                if plugin.data._namespace not in store_namespaces:
                    store_namespaces.add(plugin.data._namespace)
                    await store_manager.register_model(plugin.data)
                else:
                    raise ValueError(f"Plugin {name} data namespace {plugin.data._namespace} is duplicated.")


        def create_model(model_name: str, enable_thinking: bool = False):
            splited_model_name = model_name.split(':', 1)
            if len(splited_model_name) != 2:
                raise ValueError(f"Invalid model name: {model_name}")
            else:
                provider = splited_model_name[0]
                model = splited_model_name[1]
            kwargs = {}
            if (
                'deepseek-v3.2' in model or
                'glm' in model or
                'kimi-k2.5' in model
            ):
                kwargs['reasoning_keep_policy'] = 'current'
            elif 'mimo-v2-flash' in model:
                kwargs['reasoning_keep_policy'] = 'all'
            if provider == 'dashscope':
                if model.startswith(('qwen-', 'qwen3-')):
                    return ChatQwen(
                        model=model,
                        enable_thinking=enable_thinking
                    )
                elif model.startswith(('qwq-', 'qvq-')):
                    return ChatQwQ(
                        model=model
                    )
                else:
                    if enable_thinking:
                        kwargs['extra_body'] = {"enable_thinking": True}
                    return load_chat_model(
                        model=model_name,
                        **kwargs,
                    )
            if provider == 'openrouter':
                kwargs['extra_body'] = {'reasoning': {'enabled': enable_thinking}}
            else:
                kwargs['extra_body'] = {"thinking": {"type": "enabled" if enable_thinking else "disabled"}}
            return load_chat_model(
                model=model_name,
                **kwargs
            )

        chat_enable_thinking = os.getenv("CHAT_MODEL_ENABLE_THINKING", '').lower()
        if chat_enable_thinking == "true":
            chat_enable_thinking = True
        else:
            chat_enable_thinking = False
        self.chat_model = create_model(os.getenv("CHAT_MODEL_NAME", ""), chat_enable_thinking)

        structured_enable_thinking = os.getenv("STRUCTURED_MODEL_ENABLE_THINKING", '').lower()
        if structured_enable_thinking == "true":
            structured_enable_thinking = True
        else:
            structured_enable_thinking = False
        self.structured_model = create_model(os.getenv("STRUCTURED_MODEL_NAME", ""), structured_enable_thinking)

        self.main_graph = await MainGraph.create(
            llm=self.chat_model,
            plugins_with_name=self.plugins_with_name,
            llm_for_structured_output=self.structured_model)
        self.main_graph_state_merger = StateMerger(MainState)

        for plugin in self.plugins_with_name.values():
            await plugin.on_manager_init()

        # 启动heartbeat
        for agent_id in get_init_on_startup_agent_ids():
            await self.init_agent(agent_id)
        if self.heartbeat_task is None:
            self.heartbeat_task = asyncio.create_task(self.start_heartbeat_task())


    async def start_heartbeat_task(self):
        if self.heartbeat_is_running and self.heartbeat_task is not None:
            return
        self.heartbeat_is_running = True
        while self.heartbeat_is_running:
            self.on_heartbeat_finished.clear()
            await self.trigger_agents()
            self.on_heartbeat_finished.set()
            await asyncio.sleep(self.heartbeat_interval)
        logger.info("Heartbeat task stopped.")


    async def trigger_agents(self):
        """trigger所有agent，如果上一次trigger_agents正在运行则跳过"""
        if self.on_trigger_agents_finished.is_set():
            self.on_trigger_agents_finished.clear()
            await tick_schedules()
            # 这里暂时看起来有些奇怪，之后会考虑把trigger_agent放到tick_schedules中
            tasks = [self.trigger_agent(agent_id) for agent_id in self.activated_agent_id_datas.keys()]
            await asyncio.gather(*tasks)
            self.on_trigger_agents_finished.set()

    async def trigger_agent(self, agent_id: str):
        """trigger单一agent，如果上一次trigger_agent正在运行则跳过"""
        if agent_id not in self.activated_agent_id_datas:
            logger.warning(f"Agent {agent_id} 没有在activated_agent_ids中找到，说明存在非法的trigger_agent调用，需检查代码。")
            return
        if not self.activated_agent_id_datas[agent_id]['on_trigger_finished'].is_set():
            return
        self.activated_agent_id_datas[agent_id]['on_trigger_finished'].clear()

        try:
            config = {"configurable": {"thread_id": agent_id}}
            agent_store = await store_manager.get_builtin(agent_id)
            agent_settings = agent_store.settings
            agent_states = agent_store.states

            # 获取时间
            current_times = Times.from_time_settings(agent_settings.main.time_settings)

            # 如果是首次运行，则添加或发送引导消息
            if agent_states.is_first_time:
                agent_states.is_first_time = False
                if self.main_graph.is_agent_running(agent_id):
                    logger.warning(f"Agent {agent_id} 已被调用，将跳过引导消息。")
                else:
                    instruction_message = construct_system_message(
                        f'''当前时间是：{format_time(current_times.agent_world_datetime)}。
这是你被初始化以来的第一条消息。如果你看到这条消息，说明在此消息之前你还没有收到过任何来自用户的消息。
这意味着你的“记忆”暂时是空白的，如果检索记忆时提示“没有找到任何匹配的记忆。”或检索不到什么有用的信息，这是正常的。
接下来是你与用户的初次见面，请根据你所扮演的角色以及以下的提示考虑应做出什么反应：\n''' + agent_settings.main.instruction_prompt,
                        current_times
                    )
                    if agent_settings.main.react_instruction:
                        await self.call_agent_for_system(agent_id, instruction_message.text, times=current_times)
                    else:
                        instruction_messages = [instruction_message]
                        if agent_settings.main.initial_ai_messages:
                            instruction_messages.extend(
                                random.choice(agent_settings.main.initial_ai_messages)
                                .construct_messages(current_times)
                            )
                        await self.main_graph.update_messages(agent_id, instruction_messages)


            # 处理每天的任务
            last_update_agent_world_datetime = agent_states.last_updated_times.agent_world_datetime
            if (
                current_times.agent_world_datetime.day != last_update_agent_world_datetime.day or
                current_times.agent_world_datetime.month != last_update_agent_world_datetime.month or
                current_times.agent_world_datetime.year != last_update_agent_world_datetime.year
            ):
                # 处理年龄（TODO:我觉得年龄应该靠自己想，而非程序计算）
                if agent_settings.main.character_settings.birthday is not None:
                    age = relativedelta(current_times.agent_world_datetime.date(), agent_settings.main.character_settings.birthday).years
                    if age != agent_settings.main.character_settings.age:
                        agent_settings.main.character_settings.age = age

            # 更新最后更新时间
            agent_states.last_updated_times = current_times

            # 如果agent已有调用，取消以下任务
            if self.main_graph.is_agent_running(agent_id):
                return

            # 自动清理被动检索
            passive_retrieval_ttl = agent_settings.retrieval.passive_retrieval_ttl
            if passive_retrieval_ttl > 0:
                passive_retrieval_messages_to_remove = []
                for m in await self.main_graph.get_messages(agent_id):
                    bh_message_metadata = BHMessageMetadata.parse(m)
                    if (
                        bh_message_metadata.message_type == 'bh:passive_retrieval' and
                        passive_retrieval_ttl >= abs(current_times.agent_subjective_tick - bh_message_metadata.creation_times.agent_subjective_tick)
                    ):
                        passive_retrieval_messages_to_remove.append(RemoveMessage(id=m.id))
                if passive_retrieval_messages_to_remove:
                    await self.main_graph.update_messages(agent_id, passive_retrieval_messages_to_remove)

            main_graph_state = await self.main_graph.graph.aget_state(config)

            # 处理自我调用
            self_call_time_secondses = main_graph_state.values.get("self_call_time_secondses", [])
            wakeup_call_time_seconds = main_graph_state.values.get("wakeup_call_time_seconds")
            active_self_call_time_secondses_and_notes = main_graph_state.values.get("active_self_call_time_secondses_and_notes", [])
            can_call = False
            if active_self_call_time_secondses_and_notes:
                for seconds, note in active_self_call_time_secondses_and_notes:
                    if current_times.agent_world_timestampus >= seconds:
                        can_call = True
                        self_call_type = 'active'
                        break
            if self_call_time_secondses and not can_call:
                for seconds in self_call_time_secondses:
                    if current_times.agent_world_timestampus >= seconds:
                        can_call = True
                        self_call_type = 'passive'
                        break
            if wakeup_call_time_seconds and not can_call:
                if current_times.agent_world_timestampus >= wakeup_call_time_seconds:
                    can_call = True
                    self_call_type = 'passive'
            if can_call:
                await self.call_agent_for_self(agent_id, self_call_type=self_call_type)

            # 如果没有自我调用，开始尝试自动回收闲置上下文，先判断是否已超出活跃时间
            elif main_graph_state.values.get("active_time_seconds") and current_times.agent_world_timestampus > main_graph_state.values.get("active_time_seconds"):
                messages = await self.main_graph.get_messages(agent_id)
                # 最后一条消息如果为HumanMessage说明agent还没有响应
                if not isinstance(messages[-1], HumanMessage):
                    not_extracted_messages = [m for m in messages if not BHMessageMetadata.parse(m).extracted]
                    # 再次判断是否有来自用户的新消息
                    if [m for m in not_extracted_messages if BHMessageMetadata.parse(m).message_type == 'bh:user']:

                        remove_messages = []

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
                                if count_tokens_approximately(messages) > max_tokens:
                                    new_messages: list[BaseMessage] = trim_messages(
                                        messages=messages,
                                        max_tokens=max_tokens,
                                        token_counter=count_tokens_approximately,
                                        strategy='last',
                                        start_on=HumanMessage,
                                        #allow_partial=True,
                                        #text_splitter=RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
                                    )
                                    if not new_messages:
                                        logger.warning("Trim messages failed on cleanup.")
                                        new_messages = []
                                    excess_count = len(messages) - len(new_messages)
                                    old_messages = messages[:excess_count]
                                    remove_messages.extend([RemoveMessage(id=message.id) for message in old_messages])
                                    #update_messages = new_messages
                            # max_tokens <= 0 则全部删除
                            else:
                                remove_messages = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]

                        # 更新与清理
                        for m in not_extracted_messages:
                            bh_metadata = BHMessageMetadata.parse(m)
                            bh_metadata.extracted = True
                            m.additional_kwargs[BH_MESSAGE_METADATA_KEY] = bh_metadata.model_dump()
                        await self.main_graph.update_messages(agent_id, not_extracted_messages + remove_messages)

                        # 若有，将reflective的思考过程加入messages
                        if recycle_results.get('reflective'):
                            await self.main_graph.update_messages(agent_id, recycle_results.get('reflective', []))

            # 闲置过久（两个星期）则关闭agent
            elif (
                agent_id in self.activated_agent_id_datas and
                current_times.real_world_timestampus > (self.activated_agent_id_datas.get(agent_id, {}).get("created_at", 0) + 1209600_000_000)
            ):
                await self.close_agent(agent_id)

        finally:
            self.activated_agent_id_datas[agent_id]["on_trigger_finished"].set()


    async def init_agent(self, agent_id: str):
        """初始化agent，若agent处于triggering则等待"""
        if agent_id in self.activated_agent_id_datas:
            await self.activated_agent_id_datas[agent_id]["on_trigger_finished"].wait()
        self.activated_agent_id_datas[agent_id] = {
            "created_at": TimestampUs.now(),
            "on_trigger_finished": asyncio.Event()
        }
        self.activated_agent_id_datas[agent_id]["on_trigger_finished"].set()
        await store_manager.init_agent(agent_id)

        # 初始化计划
        memory_updaters = await get_schedules([
            Schedule.Condition(key='agent_id', value=agent_id),
            Schedule.Condition(key='schedule_type', value='memory:updater')
        ])
        if len(memory_updaters) != 5:
            await delete_schedules([s.schedule_id for s in memory_updaters])
            await add_schedules(construct_default_memory_update_schedules(agent_id))

        await self.trigger_agent(agent_id)

        enabled_plugins = get_agent_enabled_plugin_names(agent_id)
        for name, plugin in self.plugins_with_name.items():
            if name in enabled_plugins:
                await plugin.on_agent_init(agent_id)

    async def close_agent(self, agent_id: str):
        """手动关闭agent，若agent处于triggering则等待"""
        if self.activated_agent_id_datas.get(agent_id):
            await self.activated_agent_id_datas[agent_id]['on_trigger_finished'].wait()
            del self.activated_agent_id_datas[agent_id]
        enabled_plugins = get_agent_enabled_plugin_names(agent_id)
        for name, plugin in self.plugins_with_name.items():
            # 如果插件在运行途中被禁用，则不会调用on_agent_close，这可能存在问题
            if name in enabled_plugins:
                await plugin.on_agent_close(agent_id)
        store_manager.close_agent(agent_id)

    async def close_manager(self):
        logger.info("wait for the last heartbeat to close agent manager")
        self.heartbeat_is_running = False
        if self.heartbeat_task is not None:
            await self.on_heartbeat_finished.wait()
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None

        for plugin in self.plugins_with_name.values():
            await plugin.on_manager_close()

        await self.main_graph.conn.close()
        await store_stop_listener()
        logger.info("agent manager closed")


    async def call_agent_for_user_with_command(
        self,
        agent_id: str,
        user_input: Union[str, list[str]],
        user_name: Optional[str] = None,
        is_admin: bool = False
    ):
        extracted_message = extract_text_parts(user_input)
        if extracted_message and extracted_message[0].startswith("/"):
            if is_admin:
                await self.command_processing(agent_id, extracted_message[0])
            else:
                await self.event_queue.put({
                    "agent_id": agent_id,
                    "name": "log",
                    "args": {"content": "无权限执行此命令"},
                    "id": "command-" + str(uuid4())
                })
        else:
            await self.call_agent_for_user(
                agent_id=agent_id,
                user_input=user_input,
                user_name=user_name,
            )


    async def call_agent_for_user(
        self,
        agent_id: str,
        user_input: Union[str, list[str]],
        user_name: Optional[str] = None
    ):
        # 初始化变量
        context = MainContext(agent_id=agent_id, call_type='human')
        store_settings = await store_manager.get_settings(agent_id)
        time_settings = store_settings.main.time_settings
        current_times = Times.from_time_settings(time_settings)
        formated_agent_world_time = format_time(current_times.agent_world_datetime)

        # 处理用户姓名
        if isinstance(user_name, str):
            user_name = user_name.strip()
        name = user_name or "未知姓名"
        # 加上时间信息
        if isinstance(user_input, str):
            input_content = f'[{formated_agent_world_time}]\n{name}: {user_input}'
        elif isinstance(user_input, list):
            if len(user_input) == 1:
                input_content = f'[{formated_agent_world_time}]\n{name}: {user_input[0]}'
            elif len(user_input) > 1:
                input_content = [{'type': 'text', 'text': f'[{formated_agent_world_time}]\n{name}: {message}'} for message in user_input]
            else:
                raise ValueError("Input list cannot be empty")
        else:
            raise ValueError("Invalid input type")
        graph_input = {"input_messages": HumanMessage(
            content=input_content,
            name=user_name,
            additional_kwargs={
                BH_MESSAGE_METADATA_KEY: BHMessageMetadata(
                    creation_times=current_times,
                    message_type='bh:user'
                ).model_dump()
            }
        )}

        await self.call_agent(graph_input, context, random_wait=True)

    async def call_agent_for_system(self, agent_id: str, content: str, times: Optional[Times] = None):
        context = MainContext(agent_id=agent_id, call_type='system')
        if times is None:
            store_settings = await store_manager.get_settings(agent_id)
            time_settings = store_settings.main.time_settings
            times = Times.from_time_settings(time_settings)

        graph_input = {"input_messages": construct_system_message(
            content=content,
            times=times
        )}

        await self.call_agent(graph_input, context, random_wait=False)

    async def call_agent_for_self(self, agent_id: str, self_call_type: Literal['passive', 'active', 'wakeup'] = 'passive'):
        context = MainContext(agent_id=agent_id, call_type='self', self_call_type=self_call_type)

        if self_call_type == 'active':
            await self.call_agent({}, context, random_wait=False, is_self_call=True)
        else:
            await self.call_agent({}, context, double_texting_strategy='reject', random_wait=False, is_self_call=True)

    async def call_agent(
        self,
        graph_input: dict[str, Any],
        graph_context: MainContext,
        double_texting_strategy: Literal['merge', 'enqueue', 'reject'] = 'merge',
        random_wait: bool = False,
        is_self_call: bool = False
    ):
        """如果使用enqueue，那么这一次调用实际可能调用多次agent，造成返回等待时间较长，最好使用create_task，不用等待方法返回

        如果使用enqueue或reject，那么自然也触发不了打断了（但依然有可能在还没开始调用图时被打断）"""
        agent_id = graph_context.agent_id
        agent_run_id = graph_context.agent_run_id
        config = {"configurable": {"thread_id": agent_id}}
        call_agent_kwargs = CallAgentKwargs(
            graph_input=graph_input,
            graph_context=graph_context,
            double_texting_strategy=double_texting_strategy,
            random_wait=random_wait,
            is_self_call=is_self_call
        )

        async def processing_after_call_agent(cancelled: Optional[Cancelled] = None):
            enabled_plugins = get_agent_enabled_plugin_names(agent_id)
            for name, plugin in self.plugins_with_name.items():
                # 同样的，如果插件在运行途中被禁用，则不会调用after_call_agent，这可能存在问题
                if name in enabled_plugins:
                    if cancelled:
                        await plugin.after_call_agent(
                            call_agent_kwargs,
                            cancelled
                        )
                    elif self.main_graph.agent_run_ids.get(agent_id) != agent_run_id:
                        await plugin.after_call_agent(
                            call_agent_kwargs,
                            Cancelled('interrupted')
                        )
                    else:
                        await plugin.after_call_agent(
                            call_agent_kwargs,
                            None
                        )

        if double_texting_strategy != 'reject':
            # 首先把参数加入buffer
            if agent_id not in self._call_agent_buffers:
                kwargs_index = 0
                self._call_agent_buffers[agent_id] = [call_agent_kwargs]
            else:
                kwargs_index = len(self._call_agent_buffers[agent_id])
                self._call_agent_buffers[agent_id].append(call_agent_kwargs)

        # 处理插件，支持在调用插件时就被打断
        cancelled_plugin_name = None
        self._agent_run_id_on_before_call_agent[agent_id] = agent_run_id
        enabled_plugins = get_agent_enabled_plugin_names(agent_id)
        for name, plugin in self.plugins_with_name.items():
            if name in enabled_plugins:
                if cancelled_plugin_name:
                    await plugin.before_call_agent(call_agent_kwargs, Cancelled('plugin', cancelled_plugin_name))
                else:
                    cancelled = await plugin.before_call_agent(call_agent_kwargs, None)
                    if cancelled:
                        cancelled_plugin_name = name
        if cancelled_plugin_name:
            await processing_after_call_agent(Cancelled('plugin', cancelled_plugin_name))
            return
        elif self._agent_run_id_on_before_call_agent.get(agent_id) != agent_run_id:
            await processing_after_call_agent(Cancelled('interrupted'))
            return
        else:
            del self._agent_run_id_on_before_call_agent[agent_id]

        # 如果策略不为merge的同时已有运行，则不重复运行。如果agent_run_id相同，则允许运行，这是为enqueue准备的
        if self.main_graph.agent_run_ids.get(agent_id) != agent_run_id:
            if double_texting_strategy == 'reject':
                await processing_after_call_agent(Cancelled('rejected'))
                return
            elif double_texting_strategy == 'enqueue':
                await processing_after_call_agent(Cancelled('queuing'))
                return

        # 将运行id加入main_graph的运行id字典
        self.main_graph.agent_run_ids[agent_id] = agent_run_id

        # 当前agent有gathered，说明是在chatbot节点处打断了上次运行，则将上次运行结果加入中断数据字典
        interrupt_data = self._agent_interrupt_datas.pop(agent_id, {})
        if interrupt_data.get('chunk'):
            if self.main_graph.agent_interrupt_datas.get(agent_id):
                logger.warning(f"Agent {agent_id} has interrupt data, but the last run was interrupted. The previous interrupt data will be discarded.")
            self.main_graph.agent_interrupt_datas[agent_id] = interrupt_data

        # 随机时长的等待，模拟人不会一直盯着新消息，也防止短时间的双发
        if random_wait:
            await asyncio.sleep(random.uniform(1.0, 4.0))
            # 如果在等待期间又有新的调用，则取消这次调用
            if self.main_graph.agent_run_ids.get(agent_id, '') != agent_run_id:
                await processing_after_call_agent(Cancelled('interrupted'))
                return

        # 如果main_graph正在运行"tools"节点，则等待其运行完毕再打断。
        config = {"configurable": {"thread_id": agent_id}}
        main_graph_state = await self.main_graph.graph.aget_state(config)
        if main_graph_state.next:
            # next[0]也意味着不支持多节点并行
            current_node = main_graph_state.next[0]
            while (
                current_node == "tools" or
                current_node == "tool_node_post_process" or
                current_node == "prepare_to_recycle" or
                current_node == "begin"
            ):
                await asyncio.sleep(0.2)
                # 如果在等待期间又有新的调用，则取消这次调用
                if self.main_graph.agent_run_ids.get(agent_id) != agent_run_id:
                    await processing_after_call_agent(Cancelled('interrupted'))
                    return
                main_graph_state = await self.main_graph.graph.aget_state(config)
                current_node = main_graph_state.next[0]

        # 从buffer中取出user_input
        # 默认buffer中是有数据的，除了reject不使用buffer
        if double_texting_strategy == 'merge':
            graph_inputs = [kwargs.graph_input for kwargs in self._call_agent_buffers.pop(agent_id)]
            graph_input = self.main_graph_state_merger.merge(graph_inputs)
        elif double_texting_strategy == 'enqueue':
            # 把自己刚存进去的数据拿出来
            graph_input = self._call_agent_buffers[agent_id].pop(kwargs_index).graph_input
        elif self._call_agent_buffers.get(agent_id):
            del self._call_agent_buffers[agent_id]
            logger.warning("在使用reject策略调用agent时意外发现存在残留未处理的用户输入，已删除。")


        first = True
        tool_index = 0
        last_message = ''
        canceled = False
        gathered = None
        streaming_tool_messages = []
        store_settings = await store_manager.get_settings(agent_id)
        async for typ, msg in self.main_graph.graph.astream(graph_input, config=config, context=graph_context, stream_mode=["updates", "messages"]):
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
                    del self._agent_interrupt_datas[agent_id]

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
                #print(msg, end="\n\n", flush=True)
                if msg[1]['langgraph_node'] != 'chatbot':
                    continue

                if canceled:
                    continue

                if not self.main_graph.agent_run_ids.get(agent_id) or self.main_graph.agent_run_ids.get(agent_id) != agent_run_id:
                    canceled = True
                    continue

                if isinstance(msg[0], AIMessageChunk):

                    chunk: AIMessageChunk = msg[0]

                    if first or chunk.id != gathered.id:
                        first = False
                        gathered = chunk
                        streaming_tool_messages = []
                    else:
                        gathered += chunk
                    if agent_id not in self._agent_interrupt_datas.keys():
                        self._agent_interrupt_datas[agent_id] = {
                            'called_tool_messages': []
                        }
                    self._agent_interrupt_datas[agent_id]['chunk'] = gathered
                    self._agent_interrupt_datas[agent_id]['last_chunk_times'] = Times.from_time_settings(store_settings.main.time_settings)

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
                                    event_item["is_self_call"] = is_self_call
                                    if not chunk_completed:
                                        event_item["not_completed"] = True
                                    else:
                                        if tool_calls[tool_index].get('id'):
                                            now_times = Times.from_time_settings(store_settings.main.time_settings)
                                            streaming_tool_messages.append(ToolMessage(
                                                content=SEND_MESSAGE_TOOL_CONTENT,
                                                name=SEND_MESSAGE,
                                                tool_call_id=tool_calls[tool_index]['id'],
                                                additional_kwargs={
                                                    BH_MESSAGE_METADATA_KEY: BHMessageMetadata(
                                                        creation_times=now_times,
                                                        message_type='bh:tool'
                                                    ).model_dump()
                                                }
                                            ))
                                            self._agent_interrupt_datas[agent_id]["called_tool_messages"] = streaming_tool_messages
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
                                        now_times = Times.from_time_settings(store_settings.main.time_settings)
                                        streaming_tool_messages.append(ToolMessage(
                                            content=result,
                                            name=tool_calls[tool_index]['name'],
                                            tool_call_id=tool_calls[tool_index]['id'],
                                            additional_kwargs={
                                                BH_MESSAGE_METADATA_KEY: BHMessageMetadata(
                                                    creation_times=now_times,
                                                    message_type='bh:tool'
                                                ).model_dump()
                                            }
                                        ))
                                        self._agent_interrupt_datas[agent_id]["called_tool_messages"] = streaming_tool_messages
                                    await self.event_queue.put({"agent_id": agent_id, "name": tool_calls[tool_index]['name'], "args": tool_calls[tool_index]['args'], "id": tool_call_id})
                                    #print(await method(tool_calls[tool_index]['args']), flush=True)
                                tool_index += 1
                                loop_once = True

        await processing_after_call_agent()

        if self.main_graph.agent_run_ids.get(agent_id) == agent_run_id:
            if self._call_agent_buffers.get(agent_id):
                kwargs = self._call_agent_buffers[agent_id][0]
                if kwargs.double_texting_strategy == 'enqueue':
                    del self._call_agent_buffers[agent_id][0]
                    self.main_graph.agent_run_ids[agent_id] = kwargs.graph_context.agent_run_id
                    await self.call_agent(**kwargs.model_dump())
                else:
                    logger.warning('当call_agent运行完毕且agent_run_id没有改变时，该agent_id的_call_agent_buffer中不应会出现策略非enqueue（也就是merge，reject不会使用buffer）的调用。也许有极小概率两次调用刚好擦肩而过，总之这里就将其忽略了')
                return
            else:
                del self.main_graph.agent_run_ids[agent_id]
        return


    async def set_agent_time_settings(self, agent_id: str, new_time_settings: AgentTimeSettings):
        """设置agent的时间设置

        这个专门的方法是为了在设置新的时间设置的同时，自动更新所有可能受此时间设置影响的schedule"""
        store_settings = await store_manager.get_settings(agent_id)
        current_time_settings = store_settings.main.time_settings
        current_times = Times.from_time_settings(current_time_settings)
        new_times = Times.from_time_settings(new_time_settings, current_times)

        if new_times.agent_subjective_tick < current_times.agent_subjective_tick:
            logger.warning(f'agent {agent_id} 正在将主观tick倒流，这个操作的语义相当于要使整个系统往后倒退，而这是不可能的，所以应尽量避免这种不合理的操作发生。')
        store_settings.main.time_settings = new_time_settings
        # 如果agent_world时间倒流
        if new_times.agent_world_timestampus < current_times.agent_world_timestampus:
            outdated_schedules = await get_schedules([
                Schedule.Condition(key='agent_id', value=agent_id),
                Schedule.Condition(key='time_reference', value='agent_world'),
                Schedule.Condition(key='repeating', value=True)
            ])
            ids_to_delete = []
            new_values_to_update = []
            for schedule in outdated_schedules:
                try:
                    new_values = schedule.calc_trigger_time(new_times)
                    if new_values is None:
                        ids_to_delete.append(schedule.schedule_id)
                    else:
                        new_values_to_update.append(new_values)
                # 忽略，这意味着计划的触发时间在新的时间下也没有改变，所以不需要任何操作
                except Schedule.SameTimeError:
                    pass
            if ids_to_delete:
                await delete_schedules(ids_to_delete)
            if new_values_to_update:
                await update_schedules(new_values_to_update)


    async def command_processing(self, agent_id: str, user_input: str):
        async def _command_processing(agent_id: str, user_input: str) -> str:
            config = {"configurable": {"thread_id": agent_id}}

            if user_input == "/help":
                return """可用指令列表：
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
/skip_agent_time <world|subjective> <时间> - 跳过agent时间

使用 /<指令> help 查看具体使用说明"""
            elif user_input.startswith("/get_state ") or user_input == "/get_state":
                if user_input == "/get_state help":
                    return """使用方法：/get_state <key>
key: 要获取的状态键名，例如 /get_state active_time_seconds"""
                elif user_input != "/get_state":
                    splited_input = user_input.split(" ")
                    requested_key = splited_input[1]

                    state = await self.main_graph.graph.aget_state(config)
                    if requested_key == 'next':
                        return f"状态[{requested_key}]: {state.next or '未运行'}"
                    return f"状态[{requested_key}]: {state.values.get(requested_key, '未找到该键')}"

            elif user_input.startswith("/delete_last_messages ") or user_input == "/delete_last_messages":
                if user_input == "/delete_last_messages help":
                    return """使用方法：/delete_last_messages <数量>
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
                            return f"已删除最后{len(remove_messages)}条消息。"
                        else:
                            return "没有找到任何消息"

            elif user_input.startswith("/set_role_prompt ") or user_input == "/set_role_prompt":
                if user_input == "/set_role_prompt help":
                    return """使用方法：/set_role_prompt <提示词>
提示词: 要设置的角色提示词，例如 /set_role_prompt 你是一个友好的助手"""
                elif user_input != "/set_role_prompt":
                    splited_input = user_input.split(" ", 1)
                    role_prompt = splited_input[1].strip()
                    if role_prompt:
                        agent_settings = await store_manager.get_settings(agent_id)
                        agent_settings.main.role_prompt = role_prompt
                        return "角色提示词设置成功"
                    else:
                        return "角色提示词不能为空"

            elif user_input == "/load_config" or user_input.startswith("/load_config "):
                if user_input == "/load_config help":
                    return """使用方法：/load_config [agentID|__all__]
agentID: 要加载的特定agent配置
__all__: 加载所有agent配置
例如：/load_config agent_1 或 /load_config __all__"""
                else:
                    if user_input == "/load_config":
                        await load_config(self.plugins_with_name, agent_id, force=True)
                    else:
                        splited_input = user_input.split(" ")
                        if splited_input[1]:
                            if splited_input[1] == "__all__":
                                await load_config(self.plugins_with_name, force=True)
                            else:
                                await load_config(self.plugins_with_name, splited_input[1], force=True)
                    await store_manager.init_agent(agent_id)
                    return "配置文件已加载。"

            elif user_input == "/wakeup" or user_input == "/wakeup help":
                if user_input == "/wakeup help":
                    return """使用方法：/wakeup
唤醒agent（重置agent的自我调用状态），这可能导致agent对prompt有些误解。"""
                else:
                    await self.main_graph.graph.aupdate_state(
                        config,
                        {
                            "active_time_seconds": TimestampUs(0),
                            "self_call_time_secondses": [],
                            "wakeup_call_time_seconds": TimestampUs(0)
                        },
                        as_node='final'
                    )
                    return "已唤醒agent（重置自我调用相关状态），这可能导致agent对prompt有些误解。"

            elif user_input == "/messages" or user_input == "/messages help":
                if user_input == "/messages help":
                    return """使用方法：/messages
查看agent消息列表中的所有消息。"""
                else:
                    main_messages = await self.main_graph.get_messages(agent_id)
                    if main_messages:
                        time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
                        return format_messages_for_ai(main_messages)
                    else:
                        return "agent消息列表为空。"

            elif user_input == "/tokens" or user_input == "/tokens help":
                if user_input == "/tokens help":
                    return """使用方法：/tokens
计算agent消息列表的token数量。"""
                else:
                    main_messages = await self.main_graph.get_messages(agent_id)
                    if main_messages:
                        return str(count_tokens_approximately(main_messages))
                    else:
                        return "agent消息列表为空，无法计算token数量。"

            elif user_input.startswith("/memories ") or user_input == "/memories":
                if user_input == "/memories help":
                    return """使用方法：/memories <类型> [偏移] [数量]
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

ttl: {get_result["metadatas"][i]["ttl"]}

retrievability: {get_result["metadatas"][i]["retrievability"]}''' for i in range(len(get_result["ids"]))])
                    if not message:
                        return "没有找到任何记忆。"
                    else:
                        return message

            elif user_input == "/reset" or user_input.startswith("/reset "):
                if user_input == "/reset help":
                    return """使用方法：/reset <all|config>
all: 重置该agent所有数据（运行时不可用）
config: 仅重置配置（settings）
例如：/reset config 或 /reset all"""
                elif user_input != "/reset":
                    splited_input = user_input.split(" ")
                    if len(splited_input) >= 2 and splited_input[1]:
                        reset_type = splited_input[1]
                        if reset_type == 'config':
                            await store_adelete_namespace((agent_id, 'model', 'settings'))
                            await store_manager.init_agent(agent_id)
                            return "已重置该agent配置。"
                        elif reset_type == 'all':
                            if not self.main_graph.is_agent_running(agent_id):
                                await self.close_agent(agent_id)
                                agent_schedules = await get_schedules([
                                    Schedule.Condition(key='agent_id', value=agent_id)
                                ])
                                await delete_schedules(agent_schedules)
                                await self.main_graph.graph.checkpointer.adelete_thread(agent_id)
                                memory_manager.delete_collection(agent_id, "original")
                                memory_manager.delete_collection(agent_id, "episodic")
                                memory_manager.delete_collection(agent_id, "reflective")
                                await store_adelete_namespace(('agents', agent_id))
                                await load_config(self.plugins_with_name, agent_id)
                                await self.init_agent(agent_id)
                                return "已重置该agent所有数据。"
                            else:
                                return "agent运行时无法重置所有数据。"

            elif user_input == "/skip_agent_time" or user_input.startswith("/skip_agent_time "):
                if user_input == "/skip_agent_time help":
                    return """使用方法：/skip_agent_time <world|subjective> <时间>
类型：世界时间（world）或主观tick（subjective）
时间: 要跳过的时间（忽略时间膨胀），格式为`1w2d3h4m5s`，意为1周2天3小时4分钟5秒（也可有小数）。例如 /skip_agent_time 1w1.5d，意为跳过1周加1.5天。
注意：在涉及现实时间的一些场景如网络搜索时agent可能会感到混乱"""
                elif user_input != "/skip_agent_time":
                    splited_input = user_input.split(" ", 2)
                    if len(splited_input) == 3:
                        time_type = splited_input[1]
                        delta_str = splited_input[2]
                        try:
                            parsed_microseconds = timedelta_to_microseconds(parse_timedelta(delta_str))
                        except ValueError:
                            try:
                                parsed_microseconds = int(delta_str)
                            except (ValueError, TypeError):
                                return "时间格式错误，请确认格式正确，如 1w2d3h4m5s，或输入微秒整数。"
                        store_settings = await store_manager.get_settings(agent_id)
                        time_settings = store_settings.main.time_settings
                        if time_type == 'world':
                            new_time_settings = time_settings.add_offset_from_now(parsed_microseconds, 'world')
                            await self.set_agent_time_settings(agent_id, new_time_settings)
                        elif time_type == 'subjective':
                            new_time_settings = time_settings.add_offset_from_now(parsed_microseconds, 'subjective')
                            await self.set_agent_time_settings(agent_id, new_time_settings)
                        else:
                            return "无效的时间类型。"
                        return f"已使agent时间跳过了{format_duration(parsed_microseconds)}。"

            elif user_input == '/list_agent_reminders' or user_input.startswith("/list_agent_reminders "):
                if user_input == '/list_agent_reminders help':
                    return """使用方法：/list_agent_reminders
返回：该agent已设置的所有提醒事项"""
                elif user_input == '/list_agent_reminders':
                    schedules = await get_schedules([
                        Schedule.Condition(key='agent_id', value=agent_id),
                        Schedule.Condition(key='schedule_type', value='agent_reminder:reminder')
                    ])
                    time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
                    if schedules:
                        return f"该agent已设置且还在生效的提醒事项有：\n\n{'\n\n'.join(
                            [f'''提醒事项标题：{schedule.job_kwargs['title']}
提醒事项描述：{schedule.job_kwargs['description']}
{schedule.format_schedule(time_settings.time_zone, prefix='提醒事项', include_id=True, include_type=False)}''' for schedule in schedules]
                        )}"
                    else:
                        return "该agent目前没有设置任何提醒事项。"

            return '无效命令。'

        message = await _command_processing(agent_id, user_input)
        await self.event_queue.put({"agent_id": agent_id, "name": "log", "args": {"content": message}, "id": 'command-' + str(uuid4())})


agent_manager = AgentManager.__new__(AgentManager)
