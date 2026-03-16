from typing import Optional, Union, Any, Callable
from collections.abc import Coroutine
import os
import signal
import asyncio
from loguru import logger
from uuid import uuid4
import random

from langchain_qwq import ChatQwen, ChatQwQ
from langchain_core.messages import AIMessageChunk, HumanMessage, RemoveMessage, BaseMessage, AIMessage, AnyMessage, ToolMessage, ContentBlock
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.utils import count_tokens_approximately

from langchain_dev_utils.chat_models import load_chat_model

from become_human.types.main import InterruptData, MainState
from become_human.types.manager import CallSpriteRequest, DoubleTextingStrategy, SpriteOutput
from become_human.graphs.base import StateMerger
from become_human.graphs.main import MainGraph, SEND_MESSAGE_TOOL_CONTENT
from become_human.config import load_config, get_init_on_startup_sprite_ids, get_sprite_enabled_plugin_names
from become_human.utils import is_valid_json, gather_safe
from become_human.times import format_time, format_duration, Times, parse_timedelta, SpriteTimeSettings, timedelta_to_microseconds, TimestampUs
from become_human.message import (
    format_messages,
    extract_text_parts,
    construct_system_message,
    SpritedMsgMeta,
    SpritedMsgMetaOptionalTimes,
    add_messages,
    DEFAULT_TOOL_MSG_TYPE,
    DEFAULT_USER_MSG_TYPE
)
from become_human.store.base import store_setup, store_stop_listener, store_adelete_namespace
from become_human.store.settings import SpritedSettings
from become_human.store.states import SpritedStates
from become_human.store.manager import store_manager
from become_human.tools.send_message import SEND_MESSAGE, SEND_MESSAGE_CONTENT
from become_human.scheduler import get_schedules, tick_schedules, delete_schedules, init_schedules_db, Schedule, update_schedules
from become_human.plugin import *
from become_human.plugin import ChangeableField
from become_human.event import event_bus, ON_SPRITE_OUTPUT_EVENT


class SpriteManager:
    """sprite管理器

    sprite = agent = thread

    有些地方会出现thread、thread_id，指langgraph checkpointer的thread，在这里就被当作sprite/agent。"""

    #event_queue: asyncio.Queue
    plugins_with_name: dict[str, BasePlugin]

    activated_sprite_id_datas: dict[str, dict[str, Any]]
    heartbeat_interval: float
    heartbeat_is_running: bool
    heartbeat_task: Optional[asyncio.Task]

    # 两者目前来说是一样的
    on_heartbeat_finished: asyncio.Event
    on_trigger_sprites_finished: asyncio.Event

    chat_model: BaseChatModel
    structured_model: BaseChatModel
    main_graph: MainGraph
    main_graph_state_merger: StateMerger
    # 存储所有正在运行的call_sprite_and_wait任务，用于在close时处理未完成的任务
    _tasks: list[asyncio.Task]
    # 缓冲用于当双发但还没调用graph时，最后一次调用可以连上之前的输入给sprite，而前面的调用直接取消即可。
    _call_sprite_buffers: dict[str, list[CallSpriteRequest]]
    _sprite_interrupt_datas: dict[str, InterruptData]
    _sprite_run_id_on_before_call_sprite: dict[str, str]
    _call_sprite_buffers_on_before_call_sprite: dict[str, list[CallSpriteRequest]]
    # 给keep_input_messages用的
    _update_only_sprite_run_ids: dict[str, list[str]]
    # 以下两个都是给merge用的
    _merging_messages: dict[str, list[BaseMessage]]
    _not_streamed_graph_runs: list[str]

    standalone_loop: asyncio.AbstractEventLoop

    def run_standalone(self, plugins: Optional[list[Union[type[BasePlugin], BasePlugin]]] = None, heartbeat_interval: float = 5.0):
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is not None:
            raise RuntimeError("Standalone event loop cannot be run in an existing event loop")

        self.standalone_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.standalone_loop)

        # event或run_forever到底哪个更安全？call_soon到底有没有用？
        # 又或者轮询是最好的
        # 不知道，先这么用着
        shutdown_event = asyncio.Event()
        def shutdown(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            #self.standalone_loop.call_soon_threadsafe(self.standalone_loop.stop)
            self.standalone_loop.call_soon_threadsafe(shutdown_event.set)

        for sig in (signal.SIGTERM, signal.SIGINT):
            # Windows用不了add_signal_handler，只能用signal.signal
            #self.standalone_loop.add_signal_handler(sig, shutdown, sig)
            signal.signal(sig, shutdown)

        self.standalone_loop.create_task(self.init_manager(plugins, heartbeat_interval))
        try:
            #self.standalone_loop.run_forever()
            self.standalone_loop.run_until_complete(shutdown_event.wait())
        # 靠捕获KeyboardInterrupt是不靠谱的
        # 或者说在我这里不止是不靠谱，是会百分百报错的
        # 需要用signal，只不过Windows用不了事件循环提供的接口
        except KeyboardInterrupt:
            logger.warning("在接管了SIGINT信号不应该出现KeyboardInterrupt")
        finally:
            try:
                self.standalone_loop.run_until_complete(self.close_manager())
            except Exception:
                logger.exception(f"Error while closing standalone event loop")

            # 获取所有待处理的任务
            pending = asyncio.all_tasks(self.standalone_loop)
            if pending:
                logger.info(f"正在取消 {len(pending)} 个未完成的任务...")
                for task in pending:
                    task.cancel()

                # 等待取消操作完成
                # 使用 run_until_complete 等待 gather，如果列表为空则立即返回
                if pending:
                    results = self.standalone_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    # 打印取消结果
                    for i, result in enumerate(results):
                        if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                            logger.opt(exception=result).error(f"任务 {i} 取消时出错")

            try:
                self.standalone_loop.run_until_complete(self.standalone_loop.shutdown_asyncgens())
            except Exception:
                logger.exception(f"Error while shutting down async generators")
            self.standalone_loop.close()
            del self.standalone_loop

    def close_standalone(self):
        if getattr(self, "standalone_loop", None) is not None:
            if self.standalone_loop.is_running():
                self.standalone_loop.call_soon_threadsafe(self.standalone_loop.stop)
        else:
            raise RuntimeError("Standalone event loop is not existing")


    def __init__(self):
        """
        不要直接实例化此类。
        请导入 sprite_manager 实例变量，再调用实例方法 init_manager 完成初始化。
        """
        raise NotImplementedError("""不要直接实例化此类！
请导入 sprite_manager 实例变量，再调用实例方法 init_manager 完成初始化。""")


    async def init_manager(self, plugins: Optional[list[Union[type[BasePlugin], BasePlugin]]] = None, heartbeat_interval: float = 5.0):
        logger.info("Initializing sprite manager...")

        req_envs = ["CHAT_MODEL_NAME", "STRUCTURED_MODEL_NAME"]
        for e in req_envs:
            if not os.getenv(e):
                raise Exception(f"{e} is not set")

        #self.event_queue = asyncio.Queue()

        self.heartbeat_interval = heartbeat_interval
        self.activated_sprite_id_datas = {}
        self.heartbeat_is_running = False
        self.heartbeat_task = None

        self.on_heartbeat_finished = asyncio.Event()
        self.on_heartbeat_finished.set()
        self.on_trigger_sprites_finished = asyncio.Event()
        self.on_trigger_sprites_finished.set()

        self._tasks = []
        self._call_sprite_buffers = {}
        self._sprite_interrupt_datas = {}
        self._sprite_run_id_on_before_call_sprite = {}
        self._call_sprite_buffers_on_before_call_sprite = {}
        self._update_only_sprite_run_ids = {}
        self._merging_messages = {}
        self._not_streamed_graph_runs = []


        await store_setup()
        await init_schedules_db()

        if plugins is None:
            plugins = []
        # 按优先级排序
        plugins = PluginPriority.sort_plugins_by_priority(plugins)
        plugins_with_name = {}
        for plugin in plugins:
            # name必须是类属性
            if plugin.name not in plugins_with_name:
                plugins_with_name[plugin.name] = plugin() if isinstance(plugin, type) else plugin
            else:
                raise ValueError(f"Plugin name {plugin.name} is duplicated.")
        self.plugins_with_name = plugins_with_name

        # 检查依赖
        PluginRelation.check_relations(self.plugins_with_name)

        await load_config(self.plugins_with_name)

        config_namespaces = set([SpritedSettings._namespace])
        data_namespaces = set([SpritedStates._namespace])
        for name, plugin in self.plugins_with_name.items():
            if hasattr(plugin, 'config'):
                if plugin.config._namespace not in config_namespaces:
                    config_namespaces.add(plugin.config._namespace)
                    await store_manager.register_model(plugin.config)
                else:
                    raise ValueError(f"Plugin {name} config namespace {plugin.config._namespace} is duplicated.")
            if hasattr(plugin, 'data'):
                if plugin.data._namespace not in data_namespaces:
                    data_namespaces.add(plugin.data._namespace)
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

        event_bus.set_initialized()

        for sprite_id in get_init_on_startup_sprite_ids():
            await self.init_sprite(sprite_id)

        # 启动heartbeat
        if self.heartbeat_task is None:
            self.heartbeat_task = asyncio.create_task(self.start_heartbeat_task())
            def on_heartbeat_task_done(future: asyncio.Task):
                try:
                    future.result()
                except asyncio.CancelledError:
                    pass
            self.heartbeat_task.add_done_callback(on_heartbeat_task_done)

        logger.info("sprite manager initialized.")


    async def start_heartbeat_task(self):
        if self.heartbeat_is_running and self.heartbeat_task is not None:
            return
        self.heartbeat_is_running = True
        try:
            while self.heartbeat_is_running:
                self.on_heartbeat_finished.clear()
                await self.trigger_sprites()
                self.on_heartbeat_finished.set()
                await asyncio.sleep(self.heartbeat_interval)
        finally:
            self.heartbeat_is_running = False
            self.on_heartbeat_finished.set()
            logger.info("Heartbeat task stopped.")


    async def trigger_sprites(self):
        """trigger所有sprite，如果上一次trigger_sprites正在运行则跳过"""
        if self.on_trigger_sprites_finished.is_set():
            self.on_trigger_sprites_finished.clear()
            try:
                # 这里暂时看起来有些奇怪，之后会考虑把trigger_sprite放到tick_schedules中
                tasks = [self.trigger_sprite(sprite_id) for sprite_id in self.activated_sprite_id_datas.keys()]
                await gather_safe(*tasks)
                await tick_schedules(self.activated_sprite_id_datas.keys())
            finally:
                self.on_trigger_sprites_finished.set()

    async def trigger_sprite(self, sprite_id: str):
        """trigger单一sprite，如果上一次trigger_sprite正在运行则跳过"""
        if sprite_id not in self.activated_sprite_id_datas:
            logger.warning(f"Sprite {sprite_id} 没有在activated_sprite_ids中找到，说明存在非法的trigger_sprite调用，需检查代码。")
            return
        if not self.activated_sprite_id_datas[sprite_id]['on_trigger_finished'].is_set():
            return
        self.activated_sprite_id_datas[sprite_id]['on_trigger_finished'].clear()

        try:
            config = {"configurable": {"thread_id": sprite_id}}
            sprite_settings = store_manager.get_settings(sprite_id)
            sprite_states = store_manager.get_states(sprite_id)

            # 获取时间
            current_times = Times.from_time_settings(sprite_settings.time_settings)


            # # 处理每天的任务
            # last_update_agent_world_datetime = agent_states.last_updated_times.agent_world_datetime
            # if (
            #     current_times.agent_world_datetime.day != last_update_agent_world_datetime.day or
            #     current_times.agent_world_datetime.month != last_update_agent_world_datetime.month or
            #     current_times.agent_world_datetime.year != last_update_agent_world_datetime.year
            # ):
            #     # 处理年龄（TODO:我觉得年龄应该靠自己想，而非程序计算）
            #     if agent_settings.main.character_settings.birthday is not None:
            #         age = relativedelta(current_times.agent_world_datetime.date(), agent_settings.main.character_settings.birthday).years
            #         if age != agent_settings.main.character_settings.age:
            #             agent_settings.main.character_settings.age = age

            # 更新最后更新时间
            sprite_states.last_updated_times = current_times

            # 如果sprite已有调用，取消以下任务
            if self.is_sprite_running(sprite_id):
                return


            # 闲置过久（两个星期）则关闭sprite
            if (
                sprite_id in self.activated_sprite_id_datas and
                current_times.real_world_timestampus > (self.activated_sprite_id_datas.get(sprite_id, {}).get("created_at", 0) + 1209600_000_000)
            ):
                await self.close_sprite(sprite_id)

        finally:
            self.activated_sprite_id_datas[sprite_id]["on_trigger_finished"].set()


    async def init_sprite(self, sprite_id: str):
        """初始化sprite，若sprite处于triggering则等待"""
        if sprite_id in self.activated_sprite_id_datas:
            await self.activated_sprite_id_datas[sprite_id]["on_trigger_finished"].wait()
        self.activated_sprite_id_datas[sprite_id] = {
            "created_at": TimestampUs.now(),
            "on_trigger_finished": asyncio.Event()
        }
        await store_manager.init_sprite(sprite_id)
        self.activated_sprite_id_datas[sprite_id]["on_trigger_finished"].set()

        await self.trigger_sprite(sprite_id)

        for plugin in self.get_plugins(sprite_id):
            await plugin.on_sprite_init(sprite_id)

    async def close_sprite(self, sprite_id: str):
        """手动关闭sprite，若sprite处于triggering则等待"""
        if self.activated_sprite_id_datas.get(sprite_id):
            await self.activated_sprite_id_datas[sprite_id]['on_trigger_finished'].wait()
            del self.activated_sprite_id_datas[sprite_id]
        # 干脆直接等所有都处理完，因为这才包含了schedules
        await self.on_trigger_sprites_finished.wait()
        for plugin in self.get_plugins(sprite_id):
            # 如果插件在运行途中被禁用，则不会调用on_sprite_close，这可能存在问题
            try:
                await plugin.on_sprite_close(sprite_id)
            except Exception:
                logger.exception(f"plugin {plugin.name} on_sprite_close failed")
        store_manager.close_sprite(sprite_id)

    async def close_manager(self):
        logger.info("wait for the last heartbeat to close sprite manager")
        self.heartbeat_is_running = False
        if self.heartbeat_task is not None:
            await self.on_heartbeat_finished.wait()
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None

        if self._tasks:
            await gather_safe(*self._tasks)

        for plugin in self.plugins_with_name.values():
            try:
                await plugin.on_manager_close()
            except Exception:
                logger.exception(f"plugin {plugin.name} on_manager_close failed")

        await self.main_graph.conn.close()
        await store_stop_listener()

        try:
            standalone_loop = self.standalone_loop
        except AttributeError:
            standalone_loop = None
        self.__dict__.clear()
        if standalone_loop is not None:
            self.standalone_loop = standalone_loop

        logger.info("sprite manager closed")


    def call_sprite_for_user_with_command_nowait(
        self,
        sprite_id: str,
        user_input: Union[str, list[ContentBlock]],
        user_name: Optional[str] = None,
        is_admin: bool = False,
        **kwargs
    ) -> asyncio.Task:
        return self.add_task(self.call_sprite_for_user_with_command(
            sprite_id=sprite_id,
            user_input=user_input,
            user_name=user_name,
            is_admin=is_admin,
            **kwargs
        ))

    async def call_sprite_for_user_with_command(
        self,
        sprite_id: str,
        user_input: Union[str, list[ContentBlock]],
        user_name: Optional[str] = None,
        is_admin: bool = False,
        **kwargs
    ):
        extracted_message = extract_text_parts(user_input)
        if extracted_message and extracted_message[0].startswith("/"):
            if is_admin:
                await self.command_processing(sprite_id, extracted_message[0])
            else:
                await self.publish_sprite_output(
                    SpriteOutput(
                        sprite_id=sprite_id,
                        id="command-" + str(uuid4()),
                        extra_kwargs={
                            'log': '无权限执行此命令'
                        }
                    )
                )
        else:
            await self.call_sprite_for_user(
                sprite_id=sprite_id,
                user_input=user_input,
                user_name=user_name,
                **kwargs
            )


    def call_sprite_for_user_nowait(
        self,
        sprite_id: str,
        user_input: Union[str, list[ContentBlock]],
        user_name: Optional[str] = None,
        **kwargs
    ) -> asyncio.Task:
        return self.add_task(self.call_sprite_for_user(
            sprite_id=sprite_id,
            user_input=user_input,
            user_name=user_name,
            **kwargs
        ))

    async def call_sprite_for_user(
        self,
        sprite_id: str,
        user_input: Union[str, list[ContentBlock]],
        user_name: Optional[str] = None,
        **kwargs
    ):
        # 初始化变量
        store_settings = store_manager.get_settings(sprite_id)
        time_settings = store_settings.time_settings
        current_times = Times.from_time_settings(time_settings)
        formated_sprite_world_time = format_time(current_times.sprite_world_datetime)

        # 处理用户姓名
        if isinstance(user_name, str):
            user_name = user_name.strip()
        name = user_name or "未知姓名"
        # 加上时间信息
        if isinstance(user_input, str):
            user_input = f'[{formated_sprite_world_time}]\n{name}: {user_input}'
        elif isinstance(user_input, list):
            if not user_input:
                raise ValueError("Input list cannot be empty")
            user_input = user_input.copy()
            for i, content_block in enumerate(user_input):
                if content_block['type'] == 'text':
                    new_content_block = content_block.copy()
                    new_content_block['text'] = f'[{formated_sprite_world_time}]\n{name}: {content_block['text']}'
                    user_input[i] = new_content_block
        else:
            raise ValueError("Invalid input type")
        graph_input = [SpritedMsgMeta(
            creation_times=current_times,
            message_type=DEFAULT_USER_MSG_TYPE
        ).set_to(HumanMessage(
            content=user_input,
            name=user_name,
        ))]

        await self.call_sprite(sprite_id, graph_input, random_wait=True, **kwargs)


    def call_sprite_for_system_nowait(
        self,
        sprite_id: str,
        content: str,
        times: Optional[Times] = None,
        **kwargs
    ) -> asyncio.Task:
        return self.add_task(self.call_sprite_for_system(
            sprite_id=sprite_id,
            content=content,
            times=times,
            **kwargs
        ))

    async def call_sprite_for_system(
        self,
        sprite_id: str,
        content: str,
        times: Optional[Times] = None,
        **kwargs
    ):
        if times is None:
            store_settings = store_manager.get_settings(sprite_id)
            time_settings = store_settings.time_settings
            times = Times.from_time_settings(time_settings)

        graph_input = [construct_system_message(
            content=content,
            times=times
        )]

        await self.call_sprite(sprite_id, graph_input, double_texting_strategy='enqueue', **kwargs)


    def call_sprite_nowait(
        self,
        sprite_id: str,
        input_messages: list[BaseMessage],
        sprite_run_id: Optional[str] = None,
        double_texting_strategy: DoubleTextingStrategy = 'merge',
        random_wait: bool = False,
        **kwargs
    ) -> asyncio.Task:
        """非阻塞调用sprite，返回task"""
        return self.add_task(
            self.call_sprite(
                sprite_id=sprite_id,
                input_messages=input_messages,
                sprite_run_id=sprite_run_id,
                double_texting_strategy=double_texting_strategy,
                random_wait=random_wait,
                **kwargs
            )
        )

    async def call_sprite(
        self,
        sprite_id: str,
        input_messages: list[BaseMessage],
        sprite_run_id: Optional[str] = None,
        double_texting_strategy: DoubleTextingStrategy = 'merge',
        random_wait: bool = False,
        **kwargs
    ):
        """如果使用enqueue或reject，那么自然也触发不了打断了（但依然有可能在还没开始调用图时被打断）

        sprite当前如果正在运行，reject会丢掉消息列表，如果在还没开始调用图时被打断，也会丢掉消息列表，被插件取消调用时也无论如何不会保留消息列表"""
        if not sprite_run_id:
            sprite_run_id = str(uuid4())
        config = {"configurable": {"thread_id": sprite_id}}

        #input_messages = [m.model_copy(deep=True) for m in input_messages]

        time_settings = store_manager.get_settings(sprite_id).time_settings
        current_times = Times.from_time_settings(time_settings)
        default_meta = SpritedMsgMetaOptionalTimes(creation_times=current_times)
        for message in input_messages:
            default_meta.fill_to(message)

        call_sprite_request = CallSpriteRequest(
            sprite_id=sprite_id,
            input_messages=input_messages,
            sprite_run_id=sprite_run_id,
            double_texting_strategy=double_texting_strategy,
            random_wait=random_wait,
            extra_kwargs=kwargs
        )
        new_messages = None
        cancelled_by_plugin_in_graph = None

        async def processing_after_call_sprite(info: Optional[AfterCallSpriteInfo] = None):
            for plugin in self.get_plugins(sprite_id):
                # 同样的，如果插件在运行途中被禁用，则不会调用after_call_sprite，这可能存在问题
                if info:
                    info_value = info.model_copy(update={'new_messages': new_messages})
                elif not self.is_current_run(sprite_id, sprite_run_id):
                    info_value = AfterCallSpriteInfo(
                        cancelled=True,
                        cancelled_reason='interrupted',
                        new_messages=new_messages
                    )
                else:
                    info_value = AfterCallSpriteInfo(
                        new_messages=new_messages
                    )
                try:
                    await plugin.after_call_sprite(
                        call_sprite_request,
                        info_value
                    )
                except Exception:
                    logger.exception(f"plugin {plugin.name} after_call_sprite failed")


        # 插件只能按队列运行
        if (
            self._sprite_run_id_on_before_call_sprite.get(sprite_id) and
            self._sprite_run_id_on_before_call_sprite.get(sprite_id) != sprite_run_id
        ):
            self._call_sprite_buffers_on_before_call_sprite.setdefault(sprite_id, []).append(call_sprite_request)
            return
        self._sprite_run_id_on_before_call_sprite[sprite_id] = sprite_run_id

        before_call_sprite_info = BeforeCallSpriteInfo(
            double_texting_strategy_ctrl=ChangeableField(current=double_texting_strategy)
        )
        for name, plugin in self.get_plugins_with_name(sprite_id).items():
            control = None
            try:
                control = await plugin.before_call_sprite(
                    call_sprite_request,
                    before_call_sprite_info,
                )
            except Exception:
                logger.exception(f"plugin {plugin.name} before_call_sprite failed")
            if control:
                before_call_sprite_info = before_call_sprite_info._update_from_control(control, name)

        double_texting_strategy = before_call_sprite_info.double_texting_strategy_ctrl.current
        call_sprite_request.double_texting_strategy = double_texting_strategy


        # 插件运行后再处理buffer，且插件必须按队列运行，保证顺序
        if before_call_sprite_info.cancel_ctrl.current:
            if before_call_sprite_info.keep_input_messages_ctrl.current:
                # 先调用on_call_sprite，插件可以修改input_messages
                on_call_sprite_info = OnCallSpriteInfo(
                    is_update_messages_only=True,
                    reason_of_update_messages_only='before_call_sprite',
                    input_messages_ctrl=ChangeableField(current=input_messages)
                )
                for plugin_name, plugin in self.get_plugins_with_name(sprite_id).items():
                    control = None
                    try:
                        control = await plugin.on_call_sprite(
                            call_sprite_request,
                            on_call_sprite_info,
                        )
                    except Exception:
                        logger.exception(f"plugin {plugin.name} on_call_sprite failed")
                    if control:
                        on_call_sprite_info = on_call_sprite_info._update_from_control(control, plugin_name, sprite_id)
                input_messages = on_call_sprite_info.input_messages_ctrl.current
                call_sprite_request.input_messages = input_messages
                # 如果已有sprite运行，排队更新
                if (
                    self.is_sprite_running(sprite_id) and
                    not self.is_current_run(sprite_id, sprite_run_id)
                ):
                    # 这是用于在之后识别是否需要只更新消息列表
                    self._update_only_sprite_run_ids.setdefault(sprite_id, []).append(sprite_run_id)
                    self._call_sprite_buffers.setdefault(sprite_id, []).append(call_sprite_request)
                # 如果sprite没有在运行，那么直接更新消息列表
                else:
                    await self.update_messages(sprite_id, input_messages)
            else:
                # 不keep就直接过
                pass
        # reject和merge不使用buffer
        elif double_texting_strategy == 'interrupt':
            # 首先把参数加入buffer
            # 如果是interrupt策略，那么会尝试合并所有未取出的buffer
            buffers = self._call_sprite_buffers.get(sprite_id, [])
            if buffers:
                all_input_messages = buffers[0].input_messages
                for call_kwargs in buffers[1:]:
                    all_input_messages = add_messages(all_input_messages, call_kwargs.input_messages)
            else:
                all_input_messages = []
            self._call_sprite_buffers[sprite_id] = all_input_messages + input_messages
            self._update_only_sprite_run_ids.pop(sprite_id, None)
            kwargs_index = len(self._call_sprite_buffers[sprite_id]) - 1
        # enqueue则正常加入buffer
        elif double_texting_strategy == 'enqueue':
            self._call_sprite_buffers.setdefault(sprite_id, []).append(call_sprite_request)
            kwargs_index = len(self._call_sprite_buffers[sprite_id]) - 1
        else:
            pass

        # 如果队列中还有未处理的参数，则继续处理
        if self._call_sprite_buffers_on_before_call_sprite.get(sprite_id):
            call_sprite_request_on_before = self._call_sprite_buffers_on_before_call_sprite[sprite_id].pop(0)
            self._sprite_run_id_on_before_call_sprite[sprite_id] = call_sprite_request_on_before.sprite_run_id
            self.call_sprite_nowait(
                sprite_id=call_sprite_request_on_before.sprite_id,
                input_messages=call_sprite_request_on_before.input_messages,
                sprite_run_id=call_sprite_request_on_before.sprite_run_id,
                double_texting_strategy=call_sprite_request_on_before.double_texting_strategy,
                random_wait=call_sprite_request_on_before.random_wait,
                **call_sprite_request_on_before.extra_kwargs
            )
        else:
            del self._sprite_run_id_on_before_call_sprite[sprite_id]

        if before_call_sprite_info.cancel_ctrl.current:
            await processing_after_call_sprite(
                AfterCallSpriteInfo(
                    cancelled=True,
                    cancelled_reason='before_call_sprite',
                    cancelled_by_plugin=before_call_sprite_info.cancel_ctrl.changes[-1].plugin_name
                )
            )
            return


        # 如果策略不为merge的同时已有运行，则不重复运行。如果sprite_run_id相同，则允许运行，这是为enqueue准备的
        if (
            double_texting_strategy != 'interrupt' and
            self.is_sprite_running(sprite_id) and
            not self.is_current_run(sprite_id, sprite_run_id)
        ):
            if double_texting_strategy == 'merge':
                # 先让graph或call不要退出，等待插件运行
                self.main_graph.set_call_sprite_merging(sprite_id, sprite_run_id)
                # 处理插件
                on_call_sprite_info = OnCallSpriteInfo(
                    is_update_messages_only=True,
                    reason_of_update_messages_only='merged',
                    input_messages_ctrl=ChangeableField(current=input_messages)
                )
                for plugin_name, plugin in self.get_plugins_with_name(sprite_id).items():
                    control = None
                    try:
                        control = await plugin.on_call_sprite(call_sprite_request, on_call_sprite_info)
                    except Exception:
                        logger.exception(f"plugin {plugin.name} on_call_sprite failed")
                    if control:
                        on_call_sprite_info = on_call_sprite_info._update_from_control(control, plugin_name, sprite_id)

                current_state = await self.main_graph.graph.aget_state(config)
                if (
                    # 如果处于从set_run_id开始直到final节点前的阶段，则直接update_messages
                    self.get_current_run_id(sprite_id) in self._not_streamed_graph_runs or
                    (current_state.next and current_state.next[0] != 'final')
                ):
                    await self.update_messages(sprite_id, on_call_sprite_info.input_messages_ctrl.current)
                else:
                    # 否则在最后阶段重新调用图
                    self._merging_messages.setdefault(sprite_id, []).extend(
                        on_call_sprite_info.input_messages_ctrl.current
                    )
                # 最后放行，如果出现新的merging则不会放行
                self.main_graph.set_call_sprite_merging_done(sprite_id, sprite_run_id)
            await processing_after_call_sprite(AfterCallSpriteInfo(
                cancelled=True,
                cancelled_reason='sprite_running',
                cancelled_by_plugin=before_call_sprite_info.double_texting_strategy_ctrl.changes[-1].plugin_name
                                    if before_call_sprite_info.double_texting_strategy_ctrl.changes else None
            ))
            return


        # 将运行id加入main_graph的运行id字典
        self.main_graph.set_current_run(sprite_id, sprite_run_id)
        self._not_streamed_graph_runs.append(sprite_run_id)

        # 当前sprite有gathered，说明是在chatbot节点处打断了上次运行，则将上次运行结果加入中断数据字典
        interrupt_data = self._sprite_interrupt_datas.pop(sprite_id, {})
        if interrupt_data.get('chunk'):
            if self.main_graph.sprite_interrupt_datas.get(sprite_id):
                logger.warning(f"Sprite {sprite_id} has interrupt data, but the last run was interrupted. The previous interrupt data will be discarded.")
            self.main_graph.sprite_interrupt_datas[sprite_id] = interrupt_data

        # 随机时长的等待，模拟人不会一直盯着新消息，也防止短时间的双发
        if random_wait:
            await asyncio.sleep(random.uniform(1.0, 4.0))
            # 如果在等待期间又有新的调用，则取消这次调用
            if not self.is_current_run(sprite_id, sprite_run_id):
                self._not_streamed_graph_runs.remove(sprite_run_id)
                await processing_after_call_sprite(AfterCallSpriteInfo(
                    cancelled=True,
                    cancelled_reason='interrupted'
                ))
                return

        # 如果main_graph正在运行除"chatbot"外的其他节点，以及on_call_sprite之后到chatbot的过程中，则等待其运行完毕再打断。
        await self.main_graph.wait_until_interruptable(sprite_id)
        # 如果在等待期间又有新的调用，则取消这次调用
        if not self.is_current_run(sprite_id, sprite_run_id):
            self._not_streamed_graph_runs.remove(sprite_run_id)
            await processing_after_call_sprite(AfterCallSpriteInfo(
                cancelled=True,
                cancelled_reason='interrupted'
            ))
            return

        # 从这里开始直到chatbot，都不允许被打断
        self.main_graph.set_is_not_interruptable(sprite_id)

        # 从buffer中取出user_input
        # 默认buffer中是有数据的，除了reject和merge不使用buffer
        if double_texting_strategy not in ('reject', 'merge'):
            # 把自己刚存进去的数据拿出来
            input_messages = self._call_sprite_buffers[sprite_id].pop(kwargs_index).input_messages
        elif self._call_sprite_buffers.get(sprite_id):
            del self._call_sprite_buffers[sprite_id]
            logger.warning("在使用reject或merge策略调用sprite时意外发现存在残留未处理的用户输入，已删除。")


        # on_call_sprite
        on_call_sprite_info = OnCallSpriteInfo(
            input_messages_ctrl=ChangeableField(current=input_messages)
        )
        for plugin_name, plugin in self.get_plugins_with_name(sprite_id).items():
            control = None
            try:
                control = await plugin.on_call_sprite(call_sprite_request, on_call_sprite_info)
            except Exception:
                logger.exception(f"plugin {plugin.name} on_call_sprite failed")
            if control:
                on_call_sprite_info = on_call_sprite_info._update_from_control(control, plugin_name, sprite_id)

        input_messages = on_call_sprite_info.input_messages_ctrl.current
        call_sprite_request.input_messages = input_messages


        async def stream_graph(input_messages: list[BaseMessage]):
            nonlocal new_messages, cancelled_by_plugin_in_graph
            graph_input = {'input_messages': input_messages}
            first = True
            tool_index = 0
            last_message = ''
            canceled = False
            gathered = None
            streaming_tool_messages = []
            store_settings = store_manager.get_settings(sprite_id)
            async for typ, msg in self.main_graph.graph.astream(graph_input, config=config, context=call_sprite_request, stream_mode=["updates", "messages"]):
                if typ == "updates":
                    #print(msg)

                    try:
                        self._not_streamed_graph_runs.remove(sprite_run_id)
                    except ValueError:
                        pass

                    if msg.get("final"):
                        new_messages = msg.get("final").get("last_new_messages")

                    if msg.get("tool_node_post_process"):
                        cancelled_by_plugin_in_graph = msg.get("tool_node_post_process").get("cancelled_by_plugin")

                    if not self.is_current_run(sprite_id, sprite_run_id):
                        canceled = True

                    if not canceled and msg.get("chatbot"):
                        first = True
                        tool_index = 0
                        gathered = None
                        streaming_tool_messages = []
                        last_message = ''
                        del self._sprite_interrupt_datas[sprite_id]

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

                    if not self.is_current_run(sprite_id, sprite_run_id):
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
                        if sprite_id not in self._sprite_interrupt_datas.keys():
                            self._sprite_interrupt_datas[sprite_id] = {
                                'called_tool_messages': []
                            }
                        self._sprite_interrupt_datas[sprite_id]['chunk'] = gathered
                        self._sprite_interrupt_datas[sprite_id]['last_chunk_times'] = Times.from_time_settings(store_settings.time_settings)

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
                                        #await self.event_queue.put({"sprite_id": sprite_id, "name": "send_message", "args": {"content": new_message.replace(last_message, '', 1)}, "not_completed": True})
                                        event_item = SpriteOutput(sprite_id=sprite_id, method="send_message", params={"content": new_message}, id=tool_call_id)
                                        # 暂时用来给app的通知服务使用，如果是自我调用就推送通知
                                        event_item.extra_kwargs["is_self_call"] = kwargs.get("is_self_call", False)
                                        if not chunk_completed:
                                            event_item.extra_kwargs["not_completed"] = True
                                        else:
                                            if tool_calls[tool_index].get('id'):
                                                now_times = Times.from_time_settings(store_settings.time_settings)
                                                streaming_tool_messages.append(SpritedMsgMeta(
                                                    creation_times=now_times,
                                                    message_type=DEFAULT_TOOL_MSG_TYPE
                                                ).set_to(ToolMessage(
                                                    content=SEND_MESSAGE_TOOL_CONTENT,
                                                    name=SEND_MESSAGE,
                                                    tool_call_id=tool_calls[tool_index]['id'],
                                                )))
                                                self._sprite_interrupt_datas[sprite_id]["called_tool_messages"] = streaming_tool_messages
                                            last_message = ''
                                        await self.publish_sprite_output(event_item)
                                        #print(new_message.replace(last_message, '', 1), end="", flush=True)
                                        last_message = new_message


                                if chunk_completed:
                                    #if tool_calls[tool_index]['name'] == SEND_MESSAGE:
                                    #    last_message = ''
                                    #    await self.event_queue.put({"sprite_id": sprite_id, "name": "send_message", "args": {"content": ""}})
                                        #print('', flush=True)
                                    if hasattr(self.main_graph.streaming_tools, tool_calls[tool_index]['name']):
                                        method = getattr(self.main_graph.streaming_tools, tool_calls[tool_index]['name'])
                                        try:
                                            result = await method(tool_calls[tool_index]['args'])
                                            if tool_calls[tool_index].get('id'):
                                                now_times = Times.from_time_settings(store_settings.time_settings)
                                                streaming_tool_messages.append(SpritedMsgMeta(
                                                    creation_times=now_times,
                                                    message_type=DEFAULT_TOOL_MSG_TYPE
                                                ).set_to(ToolMessage(
                                                    content=result,
                                                    name=tool_calls[tool_index]['name'],
                                                    tool_call_id=tool_calls[tool_index]['id'],
                                                )))
                                                self._sprite_interrupt_datas[sprite_id]["called_tool_messages"] = streaming_tool_messages
                                        except Exception:
                                            logger.exception(f"calling streaming_tool {tool_calls[tool_index]['name']} failed")
                                    if tool_calls[tool_index]['name'] != SEND_MESSAGE:
                                        await self.publish_sprite_output(SpriteOutput(
                                            sprite_id=sprite_id,
                                            method=tool_calls[tool_index]['name'],
                                            params=tool_calls[tool_index]['args'],
                                            id=tool_call_id
                                        ))
                                        #print(await method(tool_calls[tool_index]['args']), flush=True)
                                    tool_index += 1
                                    loop_once = True
        await stream_graph(input_messages)


        if cancelled_by_plugin_in_graph:
            await processing_after_call_sprite(
                AfterCallSpriteInfo(
                    cancelled=True,
                    cancelled_reason='after_call_tools',
                    cancelled_by_plugin=cancelled_by_plugin_in_graph
                )
            )
        else:
            await processing_after_call_sprite()

        # 循环是为了能在更新消息后继续处理buffer
        while True:
            if self.is_current_run(sprite_id, sprite_run_id):
                if self.main_graph.is_call_sprite_merging(sprite_id):
                    # 如果正在合并，先设置不可打断，然后等待合并完成
                    self.main_graph.set_is_not_interruptable(sprite_id)
                    await self.main_graph.wait_for_call_sprite_merging(sprite_id)
                    # 检查是否是当前运行，因为等待期间可能会有其他运行（interrupt）
                    if self.is_current_run(sprite_id, sprite_run_id):
                        merged_messages = self._merging_messages.pop(sprite_id, [])
                        if merged_messages:
                            self._not_streamed_graph_runs.append(sprite_run_id)
                            await stream_graph(merged_messages)
                        else:
                            self.main_graph.set_is_interruptable(sprite_id)
                            logger.warning(f"sprite {sprite_id} call merging but no messages")
                    # 如果不是，解除不可打断状态，让新的interrupt继续执行
                    else:
                        self.main_graph.set_is_interruptable(sprite_id)
                elif self._call_sprite_buffers.get(sprite_id):
                    next_call_kwargs = self._call_sprite_buffers[sprite_id].pop(0)
                    # 如果是keep_input_messages，那么仅更新消息
                    if next_call_kwargs.sprite_run_id in self._update_only_sprite_run_ids.get(sprite_id, []):
                        self._update_only_sprite_run_ids[sprite_id].remove(next_call_kwargs.sprite_run_id)
                        await self.update_messages(
                            sprite_id,
                            next_call_kwargs.input_messages
                        )
                        continue
                    else:
                        self.main_graph.set_current_run(sprite_id, next_call_kwargs.sprite_run_id)
                        self.call_sprite_nowait(
                            sprite_id=next_call_kwargs.sprite_id,
                            input_messages=next_call_kwargs.input_messages,
                            sprite_run_id=next_call_kwargs.sprite_run_id,
                            double_texting_strategy=next_call_kwargs.double_texting_strategy,
                            random_wait=next_call_kwargs.random_wait,
                            **next_call_kwargs.extra_kwargs
                        )
                else:
                    self.main_graph.clear_current_run(sprite_id)
            break
        return

    def add_task(self, coro: Coroutine) -> asyncio.Task:
        task = asyncio.create_task(coro)
        task.add_done_callback(lambda future: future.result())
        self._tasks.append(task)
        return task


    def get_plugins_with_name(self, sprite_id: Optional[str] = None) -> dict[str, BasePlugin]:
        """获取所有插件"""
        if sprite_id:
            enabled_plugins = get_sprite_enabled_plugin_names(sprite_id)
            return {plugin_name: plugin for plugin_name, plugin in self.plugins_with_name.items() if plugin_name in enabled_plugins}
        return self.plugins_with_name

    def get_plugins(self, sprite_id: Optional[str] = None) -> list[BasePlugin]:
        """获取所有插件"""
        return list(self.get_plugins_with_name(sprite_id).values())

    def get_plugin_names(self, sprite_id: Optional[str] = None) -> list[str]:
        """获取所有插件名称"""
        return list(self.get_plugins_with_name(sprite_id).keys())

    def get_plugin(self, plugin_name: str, sprite_id: Optional[str] = None) -> BasePlugin:
        """获取插件

        Raises:
            KeyError: 如果插件不存在
        """
        if plugin_name not in self.get_plugins_with_name(sprite_id):
            raise KeyError(f'插件 {plugin_name} 不存在')
        return self.get_plugins_with_name(sprite_id)[plugin_name]

    def is_plugin_loaded(self, plugin_name: str) -> bool:
        """检查插件是否已加载"""
        return plugin_name in self.get_plugins_with_name()

    def is_plugin_enabled(self, plugin_name: str, sprite_id: str) -> bool:
        """检查插件是否已启用"""
        return plugin_name in self.get_plugins_with_name(sprite_id)

    async def get_messages(self, sprite_id: str) -> list[AnyMessage]:
        """获取sprite的消息列表"""
        return await self.main_graph.get_messages(sprite_id)

    async def update_messages(
        self,
        sprite_id: str,
        messages: list[BaseMessage],
        skip_hooks: bool = False,
    ) -> None:
        """更新sprite的消息列表"""
        await self.main_graph.update_messages(sprite_id, messages, skip_hooks)

    def is_sprite_running(self, sprite_id: str) -> bool:
        """检查sprite是否正在运行"""
        return self.main_graph.is_sprite_running(sprite_id)

    def is_current_run(self, sprite_id: str, sprite_run_id: str) -> bool:
        """检查sprite是否正在当前运行"""
        return self.main_graph.is_current_run(sprite_id, sprite_run_id)

    def get_current_run_id(self, sprite_id: str) -> Optional[str]:
        """获取sprite当前运行的id"""
        return self.main_graph.get_current_run_id(sprite_id)

    async def set_time_settings(self, sprite_id: str, new_time_settings: SpriteTimeSettings):
        """设置sprite的时间设置

        这个专门的方法是为了在设置新的时间设置的同时，自动更新所有可能受此时间设置影响的schedule"""
        store_settings = store_manager.get_settings(sprite_id)
        current_time_settings = store_settings.time_settings
        current_times = Times.from_time_settings(current_time_settings)
        new_times = Times.from_time_settings(new_time_settings, current_times)

        if new_times.sprite_subjective_tick < current_times.sprite_subjective_tick:
            logger.warning(f'sprite {sprite_id} 正在将主观tick倒流，这个操作的语义相当于要使整个系统往后倒退，而这是不可能的，所以应尽量避免这种不合理的操作发生。')
        store_settings.time_settings = new_time_settings
        # 如果sprite_world时间倒流
        if new_times.sprite_world_timestampus < current_times.sprite_world_timestampus:
            outdated_schedules = await get_schedules([
                Schedule.Condition(key='sprite_id', value=sprite_id),
                Schedule.Condition(key='time_reference', value='sprite_world'),
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


    @staticmethod
    def subscribe_sprite_output(callback: Callable) -> None:
        """订阅sprite输出事件"""
        event_bus.subscribe(ON_SPRITE_OUTPUT_EVENT, callback)

    @staticmethod
    def on_sprite_output(callback: Callable) -> Callable:
        """订阅sprite输出事件"""
        event_bus.subscribe(ON_SPRITE_OUTPUT_EVENT, callback)
        return callback

    @staticmethod
    async def publish_sprite_output(
        output: SpriteOutput
    ):
        """发布sprite输出事件"""
        await event_bus.publish(
            ON_SPRITE_OUTPUT_EVENT,
            sprite_id=output.sprite_id,
            method=output.method,
            params=output.params,
            id=output.id,
            **output.extra_kwargs
        )


    async def command_processing(self, sprite_id: str, user_input: str):
        async def _command_processing(sprite_id: str, user_input: str) -> str:
            config = {"configurable": {"thread_id": sprite_id}}

            if user_input == "/help":
                return """可用指令列表：
/help - 显示此帮助信息
/get_state <key> - 获取指定状态键值
/delete_last_messages <数量> - 删除最后几条消息
/set_role_prompt <提示词> - 设置角色提示词
/load_config [spriteID|__all__] - 加载配置
/messages - 查看所有消息
/tokens - 计算消息令牌数
/reset <all|config> - 重置sprite数据
/skip_sprite_time <world|subjective> <时间> - 跳过sprite时间

使用 /<指令> help 查看具体使用说明"""
            elif user_input.startswith("/get_state ") or user_input == "/get_state":
                if user_input == "/get_state help":
                    return """使用方法：/get_state <key>
key: 要获取的状态键名，例如 /get_state messages"""
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
                        _main_messages = await self.main_graph.get_messages(sprite_id)
                        if _main_messages:
                            _last_messages = _main_messages[-message_count:]
                            remove_messages = [RemoveMessage(id=_message.id) for _message in _last_messages if _message.id]
                            await self.main_graph.update_messages(sprite_id, remove_messages)
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
                        sprite_settings = store_manager.get_settings(sprite_id)
                        sprite_settings.role_prompt = role_prompt
                        return "角色提示词设置成功"
                    else:
                        return "角色提示词不能为空"

            elif user_input == "/load_config" or user_input.startswith("/load_config "):
                if user_input == "/load_config help":
                    return """使用方法：/load_config [spriteID|__all__]
spriteID: 要加载的特定sprite配置
__all__: 加载所有sprite配置
例如：/load_config sprite_1 或 /load_config __all__"""
                else:
                    if user_input == "/load_config":
                        await load_config(self.plugins_with_name, sprite_id, force=True)
                    else:
                        splited_input = user_input.split(" ")
                        if splited_input[1]:
                            if splited_input[1] == "__all__":
                                await load_config(self.plugins_with_name, force=True)
                            else:
                                await load_config(self.plugins_with_name, splited_input[1], force=True)
                    await store_manager.init_sprite(sprite_id)
                    return "配置文件已加载。"

#             elif user_input == "/wakeup" or user_input == "/wakeup help":
#                 if user_input == "/wakeup help":
#                     return """使用方法：/wakeup
# 唤醒agent（重置agent的自我调用状态），这可能导致agent对prompt有些误解。"""
#                 else:
#                     await self.main_graph.graph.aupdate_state(
#                         config,
#                         {
#                             "active_time_seconds": TimestampUs(0),
#                             "self_call_time_secondses": [],
#                             "wakeup_call_time_seconds": TimestampUs(0)
#                         },
#                         as_node='final'
#                     )
#                     return "已唤醒agent（重置自我调用相关状态），这可能导致agent对prompt有些误解。"

            elif user_input == "/messages" or user_input == "/messages help":
                if user_input == "/messages help":
                    return """使用方法：/messages
查看sprite消息列表中的所有消息。"""
                else:
                    main_messages = await self.main_graph.get_messages(sprite_id)
                    if main_messages:
                        return format_messages(main_messages)
                    else:
                        return "sprite消息列表为空。"

            elif user_input == "/tokens" or user_input == "/tokens help":
                if user_input == "/tokens help":
                    return """使用方法：/tokens
计算sprite消息列表的token数量。"""
                else:
                    main_messages = await self.main_graph.get_messages(sprite_id)
                    if main_messages:
                        return str(count_tokens_approximately(main_messages))
                    else:
                        return "sprite消息列表为空，无法计算token数量。"

#             elif user_input.startswith("/memories ") or user_input == "/memories":
#                 if user_input == "/memories help":
#                     return """使用方法：/memories <类型> [偏移] [数量]
# 类型: original(原始记忆), summary(记忆摘要), semantic(语义记忆)
# 偏移: 可选，从第几条开始获取，默认0
# 数量: 可选，获取多少条，默认6
# 例如：/memories original 0 3"""
#                 elif user_input != "/memories":
#                     splited_input = user_input.split(" ")
#                     memory_type = splited_input[1]
#                     limit = int(splited_input[3]) if len(splited_input) > 3 else 6
#                     offset = int(splited_input[2]) if len(splited_input) > 2 else None
#                     get_result = await memory_manager.aget(sprite_id=sprite_id, memory_type=memory_type, limit=limit, offset=offset)
#                     message = '\n\n\n'.join([f'''id: {get_result["ids"][i]}

# content: {get_result["documents"][i]}

# ttl: {get_result["metadatas"][i]["ttl"]}

# retrievability: {get_result["metadatas"][i]["retrievability"]}''' for i in range(len(get_result["ids"]))])
#                     if not message:
#                         return "没有找到任何记忆。"
#                     else:
#                         return message

            elif user_input == "/reset" or user_input.startswith("/reset "):
                if user_input == "/reset help":
                    return """使用方法：/reset <all|config>
all: 重置该sprite所有数据（运行时不可用）
config: 仅重置配置（settings）
例如：/reset config 或 /reset all"""
                elif user_input != "/reset":
                    splited_input = user_input.split(" ")
                    if len(splited_input) >= 2 and splited_input[1]:
                        reset_type = splited_input[1]
                        if reset_type == 'config':
                            await store_adelete_namespace(('sprites', sprite_id, 'configs'))
                            await store_manager.init_sprite(sprite_id)
                            return "已重置该sprite配置。"
                        elif reset_type == 'all':
                            if not self.is_sprite_running(sprite_id):
                                await self.close_sprite(sprite_id)
                                sprite_schedules = await get_schedules([
                                    Schedule.Condition(key='sprite_id', value=sprite_id)
                                ])
                                await delete_schedules(sprite_schedules)
                                for plugin in self.get_plugins(sprite_id):
                                    await plugin.on_sprite_reset(sprite_id)
                                await self.main_graph.graph.checkpointer.adelete_thread(sprite_id)
                                await store_adelete_namespace(('sprites', sprite_id))
                                await load_config(self.plugins_with_name, sprite_id)
                                await self.init_sprite(sprite_id)
                                return "已重置该sprite所有数据。"
                            else:
                                return "sprite运行时无法重置所有数据。"

            elif user_input == "/skip_sprite_time" or user_input.startswith("/skip_sprite_time "):
                if user_input == "/skip_sprite_time help":
                    return """使用方法：/skip_sprite_time <world|subjective> <时间>
类型：世界时间（world）或主观tick（subjective）
时间: 要跳过的时间（忽略时间膨胀），格式为`1w2d3h4m5s`，意为1周2天3小时4分钟5秒（也可有小数）。例如 /skip_sprite_time 1w1.5d，意为跳过1周加1.5天。
注意：在涉及现实时间的一些场景如网络搜索时sprite可能会感到混乱"""
                elif user_input != "/skip_sprite_time":
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
                        store_settings = store_manager.get_settings(sprite_id)
                        time_settings = store_settings.time_settings
                        if time_type == 'world':
                            new_time_settings = time_settings.add_offset_from_now(parsed_microseconds, 'world')
                            await self.set_time_settings(sprite_id, new_time_settings)
                        elif time_type == 'subjective':
                            new_time_settings = time_settings.add_offset_from_now(parsed_microseconds, 'subjective')
                            await self.set_time_settings(sprite_id, new_time_settings)
                        else:
                            return "无效的时间类型。"
                        return f"已使sprite时间跳过了{format_duration(parsed_microseconds)}。"

            elif user_input == '/list_reminders' or user_input.startswith("/list_reminders "):
                if user_input == '/list_reminders help':
                    return """使用方法：/list_reminders
返回：该sprite已设置的所有提醒事项"""
                elif user_input == '/list_reminders':
                    schedules = await get_schedules([
                        Schedule.Condition(key='sprite_id', value=sprite_id),
                        Schedule.Condition(key='schedule_provider', value='reminder'),
                        Schedule.Condition(key='schedule_type', value='reminder')
                    ])
                    time_settings = store_manager.get_settings(sprite_id).time_settings
                    if schedules:
                        return f"该sprite已设置且还在生效的提醒事项有：\n\n{'\n\n'.join(
                            [f'''提醒事项标题：{schedule.job_kwargs['title']}
提醒事项描述：{schedule.job_kwargs['description']}
{schedule.format_schedule(time_settings.time_zone, prefix='提醒事项', include_id=True, include_type=False)}''' for schedule in schedules]
                        )}"
                    else:
                        return "该sprite目前没有设置任何提醒事项。"

            return '无效命令。'

        message = await _command_processing(sprite_id, user_input)
        await self.publish_sprite_output(
            SpriteOutput(
                sprite_id=sprite_id,
                id='command-' + str(uuid4()),
                extra_kwargs={
                    'log': message
                }
            )
        )


sprite_manager = SpriteManager.__new__(SpriteManager)
