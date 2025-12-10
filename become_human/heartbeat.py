from typing import Optional, Any
import asyncio
from warnings import warn
from dateutil.relativedelta import relativedelta

from langchain_core.messages import HumanMessage, RemoveMessage, BaseMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from become_human.time import utcnow, agent_seconds_to_datetime, real_time_to_agent_time, datetime_to_seconds, now_seconds
from become_human.recycling import recycle_memories
from become_human.config import get_thread_configs
from become_human.store_manager import store_manager
from become_human.memory import memory_manager, get_activated_memory_types


class HeartbeatManager:

    interval: float
    thread_ids: dict[str, Any]
    is_running: bool
    task: Optional[asyncio.Task]

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.thread_ids = {}
        self.is_running = False
        self.task = None

    @classmethod
    async def create(cls, interval: float = 5.0):
        instance = cls(interval)
        await instance.start()
        return instance

    async def start(self):
        for key, value in get_thread_configs().items():
            if value.get('init_on_startup'):
                await self.init_thread(key)
        if self.task is None:
            self.task = asyncio.create_task(self.heartbeat_task())

    async def heartbeat_task(self):
        if self.is_running and self.task is not None:
            return
        self.is_running = True
        while self.is_running:
            await self.trigger_threads()
            await asyncio.sleep(self.interval)
        print("Heartbeat task stopped.")

    async def trigger_threads(self):
        tasks = [self.trigger_thread(thread_id) for thread_id in self.thread_ids.keys()]
        await asyncio.gather(*tasks)

    async def trigger_thread(self, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        store_thread = await store_manager.get_thread(thread_id)
        store_settings = store_thread.settings
        store_states = store_thread.states
        # 更新记忆
        await memory_manager.update_timers(thread_id)

        # 获取时间
        current_datetime = utcnow()
        current_time_seconds = datetime_to_seconds(current_datetime)
        current_agent_datetime = real_time_to_agent_time(current_datetime, store_settings.main.time_settings)
        current_agent_time_seconds = datetime_to_seconds(current_agent_datetime)

        # 处理每天的任务
        last_update_agent_datetime = agent_seconds_to_datetime(store_states.last_update_agent_time_seconds, store_settings.main.time_settings)
        if (current_agent_datetime.day != last_update_agent_datetime.day or
            current_agent_datetime.month != last_update_agent_datetime.month or
            current_agent_datetime.year != last_update_agent_datetime.year):

            # 处理年龄
            if store_settings.main.character_settings.birthday is not None:
                age = relativedelta(current_agent_datetime.date(), store_settings.main.character_settings.birthday).years
                if age != store_settings.main.character_settings.age:
                    store_settings.main.character_settings.age = age

        # 更新最后更新时间
        store_states.last_update_real_time_seconds = current_time_seconds
        store_states.last_update_agent_time_seconds = current_agent_time_seconds

        # 如果线程正在运行，取消以下任务
        main_graph_state = await main_graph.graph.aget_state(config)
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
            await stream_graph_updates('', thread_id, is_self_call=True, self_call_type=self_call_type)

        # 自动回收闲置上下文
        elif main_graph_state.values.get("active_time_seconds") and current_agent_time_seconds > main_graph_state.values.get("active_time_seconds"):# 已超出活跃时间
            messages = await main_graph.get_messages(thread_id)
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
                    recycles = {t: recycle_memories(t, thread_id, not_extracted_messages, llm_for_structured) for t in memory_types}
                    recycle_results = {}
                    if len(recycles) > 0:
                        graph_results = await asyncio.gather(*recycles.values())
                        recycle_results = {k: graph_results[i] for i, k in enumerate(recycles.keys())}

                    # cleanup
                    is_cleanup = store_settings.recycling.cleanup_on_non_active_recycling
                    if is_cleanup:
                        #await main_graph.update_messages(thread_id, [RemoveMessage(id=m.id) for m in messages])
                        max_tokens = store_settings.recycling.cleanup_target_size
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
                    await main_graph.update_messages(thread_id, not_extracted_messages + remove_messages)

                    # 若有，将reflective的思考过程加入messages
                    if recycle_results.get('reflective'):
                        await main_graph.update_messages(thread_id, recycle_results.get('reflective', []))

        # 闲置过久则关闭线程
        elif current_time_seconds > (self.thread_ids[thread_id]["created_at"] + 1209600):
            self.close_thread(thread_id)


    async def init_thread(self, thread_id: str):
        self.thread_ids[thread_id] = {"created_at": now_seconds()}
        await store_manager.init_thread(thread_id)
        await self.trigger_thread(thread_id)

    def close_thread(self, thread_id: str):
        if self.thread_ids.get(thread_id):
            del self.thread_ids[thread_id]
        store_manager.close_thread(thread_id)

    async def stop(self):
        print("wait for the last heartbeat to stop")
        self.is_running = False
        if self.task is not None:
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    async def stop_force(self):
        self.is_running = False
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
