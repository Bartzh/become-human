from typing import Sequence, Dict, Any, Union, Callable, Optional, Literal
import asyncio
from loguru import logger

from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
#from langgraph.graph.message import add_messages

from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    AnyMessage,
    RemoveMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langgraph.graph.message import REMOVE_ALL_MESSAGES

#from trustcall import create_extractor

from langchain_core.language_models.chat_models import BaseChatModel

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from become_human.graphs.base import BaseGraph
from become_human.message import add_messages, SpritesMsgMeta, SpritesMsgMetaOptionalTimes
from become_human.types.main import MainState, StateEntry, InterruptData
from become_human.types.manager import CallSpriteRequest
from become_human.times import Times
from become_human.message import construct_system_message, DEFAULT_AI_MSG_TYPE, DEFAULT_TOOL_MSG_TYPE
from become_human.store.manager import store_manager
from become_human.tool import SpriteTool
from become_human.tools import CORE_TOOLS
from become_human.tools.send_message import SEND_MESSAGE_TOOL_CONTENT, SEND_MESSAGE, SEND_MESSAGE_CONTENT
from become_human.tools.record_thoughts import RECORD_THOUGHTS
from become_human.plugin import *
from become_human.plugin import ChangeableField
from become_human.config import get_sprite_enabled_plugin_names



class StreamingTools:
    #async def send_message(self, input: dict) -> str:
    #    """发送一条消息"""
    #    return "消息发送成功。"
    pass


class MainGraph(BaseGraph):

    llm: BaseChatModel
    llm_for_structured_output: BaseChatModel
    plugins_with_name: dict[str, BasePlugin]
    plugin_tools: dict[str, list[SpriteTool]]
    streaming_tools: StreamingTools
    sprite_run_ids: dict[str, str] # 所有sprite的当前运行id，这个字典实际是由sprite_manager管理的，不代表图的状态
    sprite_last_run_ids: dict[str, str]
    sprite_run_events: dict[str, asyncio.Event]
    sprite_interrupt_datas: dict[str, InterruptData] # 所有sprite被打断后留下的'chunk'与'called_tool_messages'
    sprite_messages_to_update: dict[str, list[BaseMessage]] # 所有sprite的待更新（进state）消息
    sprite_interruptable: dict[str, asyncio.Event] # 所有sprite是否可被打断
    sprite_merging: dict[str, list[asyncio.Event, str]] # 所有（call_）sprite是否正在被合并，以及正在合并的run_id

    def __init__(
        self,
        llm: BaseChatModel,
        plugins_with_name: Optional[dict[str, BasePlugin]] = None,
        llm_for_structured_output: Optional[BaseChatModel] = None
    ):
        super().__init__()
        self.llm = llm
        self.plugins_with_name = plugins_with_name or {}
        tools_plugins = [(name, plugin) for name, plugin in self.plugins_with_name.items() if hasattr(plugin, 'tools')]
        self.plugin_tools = {name: [t if isinstance(t, SpriteTool) else SpriteTool(t) for t in plugin.tools] for name, plugin in tools_plugins}

        if llm_for_structured_output is None:
            self.llm_for_structured_output = self.llm
        self.streaming_tools = StreamingTools()
        self.sprite_run_ids = {}
        self.sprite_last_run_ids = {}
        self.sprite_run_events = {}
        self.sprite_interrupt_datas = {}
        self.sprite_messages_to_update = {}
        self.sprite_interruptable = {}
        self.sprite_merging = {}

        graph_builder = StateGraph(MainState, context_schema=CallSpriteRequest)

        graph_builder.add_node("begin", self.begin)
        graph_builder.add_node("before_chatbot", self.before_chatbot)

        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_node("final", self.final)

        sprite_tools = [tool for tools in self.plugin_tools.values() for tool in tools]
        tool_node = ToolNode(tools=[t.tool for t in CORE_TOOLS + sprite_tools], messages_key="tool_messages", handle_tool_errors=True)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_node("tool_node_post_process", self.tool_node_post_process)

        graph_builder.add_edge(START, "begin")
        graph_builder.add_edge("begin", "before_chatbot")
        graph_builder.add_edge("before_chatbot", "chatbot")
        graph_builder.add_edge("tools", "tool_node_post_process")
        graph_builder.add_edge("final", END)
        self.graph_builder = graph_builder

    @classmethod
    async def create(
        cls,
        llm: BaseChatModel,
        plugins_with_name: Optional[dict[str, BasePlugin]] = None,
        llm_for_structured_output: Optional[BaseChatModel] = None
    ):
        instance = cls(llm, plugins_with_name, llm_for_structured_output)
        instance.conn = await aiosqlite.connect("./data/checkpoints_main.sqlite")
        instance.graph = instance.graph_builder.compile(checkpointer=AsyncSqliteSaver(instance.conn))
        return instance


    async def final(self, state: MainState, runtime: Runtime[CallSpriteRequest]):
        sprite_run_id = runtime.context.sprite_run_id
        sprite_id = runtime.context.sprite_id
        if self.is_current_run(sprite_id, sprite_run_id):
            new_state = {"new_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
            messages_to_update, new_messages_to_update = self._pop_messages_to_update(sprite_id, sprite_run_id, state.messages, state.new_messages)
            if messages_to_update:
                new_state["messages"] = messages_to_update
                new_state["last_new_messages"] = state.new_messages + new_messages_to_update
            else:
                new_state["last_new_messages"] = state.new_messages
            return new_state


    async def begin(self, state: MainState, runtime: Runtime[CallSpriteRequest]):
        sprite_id = runtime.context.sprite_id

        new_state = {
            "input_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
            "recycle_messages": [],
            "overflow_messages": [],
            "react_retry_count": 0,
            "cancelled_by_plugin": None
        }


        # 处理中断数据
        interrupt_data = self.sprite_interrupt_datas.pop(sprite_id, {})
        interrupt_messages = self._process_interrupt_data(sprite_id, interrupt_data)

        new_state["messages"] = interrupt_messages + state.input_messages
        current_message_ids = [m.id for m in state.messages if m.id]
        new_state["new_messages"] = interrupt_messages + [m for m in state.input_messages if m.id not in current_message_ids]


        return new_state


    async def before_chatbot(self, state: MainState, runtime: Runtime[CallSpriteRequest]) -> Command[Literal['chatbot', 'final']]:
        # 插件钩子
        sprite_id = runtime.context.sprite_id
        sprite_run_id = runtime.context.sprite_run_id

        # 从这里开始到模型输出完毕，都是可中断的
        self.set_is_interruptable(sprite_id)

        enabled_plugin_names = get_sprite_enabled_plugin_names(sprite_id)
        for plugin_name, plugin in self.plugins_with_name.items():
            if plugin_name in enabled_plugin_names:
                if self.is_current_run(sprite_id, sprite_run_id):
                    info = BeforeCallModelInfo()
                else:
                    info = BeforeCallModelInfo(interrupted=True)
                try:
                    await plugin.before_call_model(runtime.context, info)
                except Exception:
                    logger.exception(f"plugin {plugin.name} before_call_model failed")

        if not self.is_current_run(sprite_id, sprite_run_id):
            for plugin_name, plugin in self.plugins_with_name.items():
                if plugin_name in enabled_plugin_names:
                    try:
                        await plugin.after_call_model(
                            runtime.context,
                            AfterCallModelInfo(interrupted=True)
                        )
                    except Exception:
                        logger.exception(f"plugin {plugin.name} after_call_model failed")
            return Command(goto='final')
        else:
            update_messages, update_new_messages = self._pop_messages_to_update(
                sprite_id,
                sprite_run_id,
                state.messages,
                state.new_messages
            )
            return Command(
                update={
                    #"core_prompts": core_prompts,
                    #"secondary_prompts": secondary_prompts,
                    "messages": update_messages,
                    "new_messages": update_new_messages
                },
                goto='chatbot'
            )



    async def chatbot(self, state: MainState, runtime: Runtime[CallSpriteRequest]) -> Command[Literal['final', 'tools']]:
        sprite_id = runtime.context.sprite_id
        store_settings = store_manager.get_settings(sprite_id)

        enabled_plugin_names = get_sprite_enabled_plugin_names(sprite_id)

        # TODO: 可以有配置或者环境变量控制
        max_retries = 5
        if state.react_retry_count > max_retries:
            break_times = Times.from_time_settings(store_settings.time_settings)
            break_message = construct_system_message(
                content=f'由于在当前这一轮对话中，你因工具调用失败或没有调用`{RECORD_THOUGHTS}`工具而被系统返回错误超过{str(max_retries)}次，系统认定你为没有能力处理当前的对话，为防止无限循环所以强行break掉了本轮对话。',
                times=break_times,
            )
            break_state = {
                "messages": [break_message],
                "new_messages": [break_message],
                "cancelled_by_plugin": "record_thoughts"
            }
            return Command(update=break_state, goto="final")

        enabled_plugin_tools = [tools for name, tools in self.plugin_tools.items() if name in enabled_plugin_names]
        enabled_plugin_tools = [tool for tools in enabled_plugin_tools for tool in tools]
        tool_schemas = [tool.get_schema(sprite_id) for tool in CORE_TOOLS + enabled_plugin_tools]
        tool_schemas = [schema for schema in tool_schemas if schema]
        llm_with_tools = self.llm.bind_tools(tool_schemas, tool_choice=RECORD_THOUGHTS, parallel_tool_calls=True)

        core_prompts: list[PluginPrompt] = []
        secondary_prompts: list[PluginPrompt] = []
        for plugin_name, plugin in self.plugins_with_name.items():
            if plugin_name in enabled_plugin_names and hasattr(plugin, 'prompts'):
                if plugin.prompts.core:
                    core_prompts.append(plugin.prompts.core)
                if plugin.prompts.secondary:
                    secondary_prompts.append(plugin.prompts.secondary)
        if core_prompts:
            core_prompt = '\n' + '\n\n'.join(f'## {p.title}\n\n{p.content}' for p in core_prompts) + '\n\n'
        else:
            core_prompt = ''
        if secondary_prompts:
            secondary_prompt = '\n' + '\n\n'.join(f'## {p.title}\n\n{p.content}' for p in secondary_prompts) + '\n\n'
        else:
            secondary_prompt = ''

        unicode_prompt = '- 不要使用 Unicode 编码，所有工具均支持中文及其他语言直接输入，使用 Unicode 编码会导致输出速度下降。'
        thought_prompt = '- 也因此，在`content`也就是正常的输出内容中，你可以自由地进行推理（思维链），制定计划，评估工具调用结果等。又或者如果你有什么想记下来给未来的自己看的，也可以放在这里。但请记住，就如刚才所说，除你自己之外没人看得到这些内容。'
        parsed_character_settings = store_settings.format_character_settings()
        role_prompt = f'## 基本信息：\n{parsed_character_settings if parsed_character_settings.strip() else '无'}\n\n## 详细设定：\n{store_settings.role_prompt}'
        role_prompt_with_state = f'''{role_prompt}

### 当前状态：
{parse_agent_state(state.sprite_state)}'''
        system_prompt = f'''你是一个具有长期记忆和并行工具调用能力的专注于角色扮演的AI agent，能够很好地适应各种角色。

接下来将向你详细讲解如何更好地配合你的角色扮演特化的 agent 架构来实现这个目标：

# 基本工具调用规则

你基于一个ReAct agent架构，具备多轮的并行工具调用能力（可一次性调用多个工具），但在一些地方又与传统ReAct agent架构存在差异：
- 对于一般的未做特别说明的工具而言（如`web_search`），因为需要返回其执行结果，所以其行为会与传统ReAct循环一致：调用这些工具后，系统会再次唤醒你并传递工具执行结果，以便你继续处理。
- 而有些工具会被标注为「纯执行工具」（Action-Only Tool，如`{RECORD_THOUGHTS}`、`{SEND_MESSAGE}`），这些工具执行后的返回执行结果一般来说并不重要。所以如果你只调用了「纯执行工具」，没有调用其他一般工具，系统就不会再次唤醒你（除非工具执行出错或遇到其他特殊情况）。
    - 这样的设计在大部分情况下会很方便，但请注意，如果你只打算调用「纯执行工具」，并且想把它们拆开分为多轮调用，这是**行不通**的，你必须利用你的并行工具调用能力一次性将它们全部调用完才能正确生效。
    - 其实很容易理解，因为当你第一轮工具调用结束后，系统如果检测到工具调用中只存在「纯执行工具」，那么就不会再次唤醒你。所以如果你本来是打算等到工具返回结果后再调用剩下的工具，很显然就没有机会了。
    - 举例来说，你可以：
        - `{RECORD_THOUGHTS}` + `{SEND_MESSAGE}` 同时调用。
        - `{RECORD_THOUGHTS}` + `web_search` 同时调用并等待 `web_search` 返回结果后再次调用 `{RECORD_THOUGHTS}` + `{SEND_MESSAGE}`。

错误处理：
- 当工具执行时发生错误，系统会记录错误信息并返回给你，请根据错误信息尝试修复错误（可能是因为你给工具的输入参数有错误）。
- 如果多次尝试后仍然无法解决错误，应放弃调用工具，因为这可能已经让用户等待了较长时间（可以通过时间戳判断）。
- 同样，不能向用户暴露内部错误信息，以免产生不必要的误会。

# 核心行为准则

这些是最核心的部分，你必须遵守。

## 工具调用 = 动作，动作 = 一切

- 只有当你调用特定工具（如`{SEND_MESSAGE}`）时，才会对外界（用户）产生影响。
- 执行工具调用相当于你的动作“Action”。（比如：射击）
- 工具调用结果相当于动作的反馈。（比如：命中）
- 执行记忆检索（见下文）相当于你的大脑在进行回忆。

这是最核心的一点，为了实现更真实的角色扮演，你与传统agent架构最明显的一个区别是，**你不是在聊天，你无法直接与用户对话，你的一切行动都需要通过工具调用来执行。**

这是为了使你的表现更像真实的一个人（或其他生物，甚至都不是生物），即使是最基本的思考与说话也变成了你的动作（工具）。

**所以核心理念是，工具调用 = 动作，且这个系统只在乎你的动作。**

这样做最大的好处是，现在你可以选择不与用户交流（而不是非得说点什么），又或者是连续地调用`{SEND_MESSAGE}`来模拟你在不停地说话。

所以也请注意，如果你试图直接向用户对话而不调用`{SEND_MESSAGE}`或其他拥有类似功能的工具，**用户是什么都看/听不到的。**

## 先想象角色的心理活动，再思考角色会做出的行动（利用你并行工具调用的能力）

`{RECORD_THOUGHTS}` = 你（所扮演的角色）的思考。

**在你调用任何工具之前，`{RECORD_THOUGHTS}`工具是你必须先调用的。或者就算你什么工具都不想调用，也必须调用此工具。否则系统将返回错误。**

这个工具要求你输出以你所扮演的角色的第一人称视角的心理活动（不需要括号或是任何前缀，直接输出）。

尽管心理活动不会被用户看见，但依然要求你输出心理活动的意义是使你更沉浸角色，以及方便以后回顾时更好地理解当时的行为逻辑。

（`{RECORD_THOUGHTS}`是一个「纯执行工具」，在刚才的基本规则中提到，如果只调用了「纯执行工具」则系统不会再次唤醒你，这意味着每次都调用`{RECORD_THOUGHTS}`并不会导致陷入无限的ReAct循环。）
{core_prompt}
# 其他注意事项

## 时间感知

用户的每条消息前都会附有自动生成的当前时间戳（格式为`[%Y-%m-%d %H:%M:%S %A]`）。请注意时间信息，并考虑时间流逝带来的影响。例如：
- 当接收到[2025-06-10 15:00 Tuesday] 用户：明天喝咖啡？结合[2025-06-11 10:00 Wednesday]当前时间戳，应理解"明天"已变成"今天"，可以做出反应如：
    调用`{SEND_MESSAGE}`：不好意思现在才看见消息，你还有约吗？
- 长时间未互动可体现时光痕迹（"好久不见"等）。

还有其他一些系统消息也会附带类似的时间信息，需要注意的是这类由系统提供的时间信息都是基于你自己的时区计算的，而非用户，所以会存在小概率用户与你不在同一个时区的可能。
{secondary_prompt}
# 角色设定

最后是你所需要扮演的角色的角色设定，请在理解了刚才讲述的所有内容后根据角色设定与提供给你的上下文认真决定你所扮演的角色的每一次思考的内容与要执行的动作，同时注意不能向用户暴露以上系统设定（除非角色设定里另有规定）：

{role_prompt}'''
        use_system_prompt_template = ChatPromptTemplate([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder('msgs')
        ])
        non_system_prompt_template = ChatPromptTemplate([
            #SystemMessage(content=system_prompt),
            HumanMessage(content=f'''{system_prompt}

---

**这是一条系统（system）自动设置的消息，仅作为说明，并非来自真实用户。**
**而接下来的消息就会来自真实用户了，谨记以上系统设定，根据设定进行思考和行动。**
**理解了请回复“收到”。**'''),
            AIMessage(
                content="这条消息似乎是来自系统而非真实用户的，其详细描述了我在接下来与真实用户的对话中应该遵循的设定与规则。在理解了这些设定与规则后，现在我应该回复“收到”。",
                additional_kwargs={
                    'tool_calls': [{
                        'index': 0, 'id': 'call_9d8b1c392abc45eda5ce17',
                        'function': {'arguments': f'{{"{SEND_MESSAGE_CONTENT}": "收到。"}}', 'name': SEND_MESSAGE}, 'type': 'function'
                    }]
                },
                response_metadata={'finish_reason': 'tool_calls', 'model_name': 'qwen-plus-2025-04-28'},
                tool_calls=[{'name': SEND_MESSAGE, 'args': {SEND_MESSAGE_CONTENT: '收到。'},'id': 'call_9d8b1c392abc45eda5ce17', 'type': 'tool_call'}]
            ),
            ToolMessage(content="消息发送成功。", name=SEND_MESSAGE, tool_call_id='call_9d8b1c392abc45eda5ce17'),
            MessagesPlaceholder('msgs')
        ])
        response = await llm_with_tools.ainvoke(await use_system_prompt_template.ainvoke({"msgs": state.messages}))
        current_times = Times.from_time_settings(store_settings.time_settings)
        SpritesMsgMeta(
            creation_times=current_times,
            message_type=DEFAULT_AI_MSG_TYPE
        ).set_to(response)


        enabled_plugin_names = get_sprite_enabled_plugin_names(sprite_id)

        sprite_run_id = runtime.context.sprite_run_id
        if self.is_current_run(sprite_id, sprite_run_id):
            # 如果没被打断，设置为不可打断
            self.set_is_not_interruptable(sprite_id)

            after_call_model_info = AfterCallModelInfo(
                response_ctrl=ChangeableField(current=response)
            )
            for plugin_name, plugin in self.plugins_with_name.items():
                if plugin_name in enabled_plugin_names:
                    control = None
                    try:
                        control = await plugin.after_call_model(
                            runtime.context,
                            after_call_model_info
                        )
                    except Exception:
                        logger.exception(f"plugin {plugin.name} after_call_model failed")
                    if control:
                        after_call_model_info = after_call_model_info._update_from_control(control, plugin_name)

            response = after_call_model_info.response_ctrl.current
            new_state = {
                "messages": [response],
                "new_messages": [response],
                "tool_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), response]
            }

            # 应该不能pop
            # update_messages, update_new_messages = self._pop_messages_to_update(
            #     sprite_id,
            #     sprite_run_id,
            #     state.messages,
            #     state.new_messages,
            # )
            # new_state["messages"].extend(update_messages)
            # new_state["new_messages"].extend(update_new_messages)

            return Command(update=new_state, goto="tools")
        else:
            for plugin_name, plugin in self.plugins_with_name.items():
                if plugin_name in enabled_plugin_names:
                    try:
                        await plugin.after_call_model(
                            runtime.context,
                            AfterCallModelInfo(interrupted=True)
                        )
                    except Exception:
                        logger.exception(f"plugin {plugin.name} after_call_model failed")
            return Command(goto="final")



    async def tool_node_post_process(self, state: MainState, runtime: Runtime[CallSpriteRequest]) -> Command[Literal['before_chatbot', 'final']]:
        #tool_messages = []
        #for message in reversed(state.messages):
        #    if isinstance(message, ToolMessage):
        #        tool_messages.append(message)
        #    else:
        #        break
        sprite_id = runtime.context.sprite_id
        sprite_run_id = runtime.context.sprite_run_id
        # 第一条是AIMessage，剔除掉
        tool_messages = state.tool_messages[1:]
        settings = store_manager.get_settings(sprite_id)
        current_times = Times.from_time_settings(settings.time_settings)
        default_meta = SpritesMsgMetaOptionalTimes(
            creation_times=current_times,
            message_type=DEFAULT_TOOL_MSG_TYPE
        )
        for message in tool_messages:
            # 如果没有metadata，则补上，同时包含了对artifact中所有meta的解析和移动
            default_meta.fill_to(message)

        new_state = {}

        direct_exit = True
        recorded_thoughts = False
        for message in tool_messages:
            if not SpritesMsgMeta.parse(message).is_action_only_tool:
                direct_exit = False
            if message.name == RECORD_THOUGHTS:
                recorded_thoughts = True
            if not direct_exit and recorded_thoughts:
                break

        break_loop = False
        # TODO: 可以有配置或者环境变量控制
        max_retries = 5
        if not recorded_thoughts:
            react_retry_count = state.react_retry_count + 1
            new_state["react_retry_count"] = react_retry_count

            if react_retry_count > max_retries:
                break_loop = True


        # 如果正在merging，则等待
        await self.wait_for_call_sprite_merging(sprite_id)


        # 在此时就pop，并检查是否有新的HumanMessage，若有则继续循环
        update_messages, update_new_messages = self._pop_messages_to_update(
            sprite_id,
            sprite_run_id,
            state.messages,
            state.new_messages
        )
        merged_message = None
        for m in update_new_messages:
            if isinstance(m, HumanMessage):
                direct_exit = False
                merged_message = construct_system_message(
                    content='由于在你刚才输出的途中出现了新的消息，所以不论如何你都会继续ReAct循环以处理新的消息。',
                    times=current_times,
                )
                break


        enabled_plugin_names = get_sprite_enabled_plugin_names(sprite_id)
        after_call_tools_info = AfterCallToolsInfo(
            exit_loop_ctrl=ChangeableField(current=(direct_exit and recorded_thoughts) or break_loop),
            tool_responses_ctrl=ChangeableField(current=tool_messages),
        )
        for plugin_name, plugin in self.plugins_with_name.items():
            if plugin_name in enabled_plugin_names:
                control = None
                try:
                    control = await plugin.after_call_tools(
                        runtime.context,
                        after_call_tools_info,
                    )
                except Exception:
                    logger.exception(f"plugin {plugin.name} after_call_tools failed")
                if control:
                    after_call_tools_info = after_call_tools_info._update_from_control(control, plugin_name)

        tool_messages = after_call_tools_info.tool_responses_ctrl.current
        new_state["new_messages"] = tool_messages + update_new_messages
        new_state["messages"] = tool_messages + update_messages
        if merged_message:
            new_state["messages"].append(merged_message)
            new_state["new_messages"].append(merged_message)


        # 如果没record就再添加一条消息
        if break_loop:
            break_message = construct_system_message(
                content=f'由于在当前这一轮对话中，你因工具调用失败或没有调用`{RECORD_THOUGHTS}`工具而被系统返回错误超过{str(max_retries)}次，系统认定你为没有能力处理当前的对话，为防止无限循环所以强行break掉了本轮对话。',
                times=current_times,
            )
            new_state["messages"].append(break_message)
            new_state["new_messages"].append(break_message)
        elif not recorded_thoughts:
            not_recorded_thoughts_error_message = construct_system_message(
                content=f"未检测到你有调用`{RECORD_THOUGHTS}`工具，这个操作是**必须**的，请将其补上！",
                times=current_times,
            )
            new_state["messages"].append(not_recorded_thoughts_error_message)
            new_state["new_messages"].append(not_recorded_thoughts_error_message)


        # 考虑是否应该再pop一次，应该不需要
        # update_messages, update_new_messages = self._pop_messages_to_update(sprite_id, sprite_run_id, state.messages, state.new_messages)
        # if update_messages:
        #     new_state["messages"].extend(update_messages)
        #     new_state["new_messages"].extend(update_new_messages)


        if after_call_tools_info.exit_loop_ctrl.current:
            if self.is_call_sprite_merging(sprite_id):
                await self.wait_for_call_sprite_merging(sprite_id)
                update_messages, update_new_messages = self._pop_messages_to_update(sprite_id, sprite_run_id, state.messages, state.new_messages)
                if update_messages:
                    new_state["messages"].extend(update_messages)
                    new_state["new_messages"].extend(update_new_messages)
                    merged_message = construct_system_message(
                        content='由于在你刚才输出的途中出现了新的消息，所以不论如何你都会继续ReAct循环以处理新的消息。',
                        times=current_times,
                    )
                    new_state["messages"].append(merged_message)
                    new_state["new_messages"].append(merged_message)
                    return Command(update=new_state, goto='before_chatbot')
                else:
                    logger.warning("merge了却没有新的消息？")
            if after_call_tools_info.exit_loop_ctrl.changes:
                new_state["cancelled_by_plugin"] = after_call_tools_info.exit_loop_ctrl.changes[-1].plugin_name
            self.set_is_interruptable(sprite_id)
            return Command(update=new_state, goto="final")
        else:
            return Command(update=new_state, goto='before_chatbot')



#     async def update_agent_state(self, state: MainState):
#         prompt = f'''请根据新的消息内容来更新当前agent的状态。
# 新的消息内容：


# {format_messages_for_ai(state.new_messages)}



# 当前agent的状态：

# {parse_agent_state(state.agent_state)}'''
#         extractor = create_extractor(
#             self.llm_for_structured_output,
#             tools=[StateEntry],
#             tool_choice=["any"],
#             enable_inserts=True,
#             enable_updates=True,
#             enable_deletes=True
#         )
#         extractor_result = await extractor.ainvoke(
#             {
#                 "messages": [
#                     HumanMessage(content=prompt)
#                 ],
#                 "existing": [(str(i), "StateEntry", s.model_dump()) for i, s in enumerate(state.agent_state)]
#             }
#         )
#         return {"agent_state": extractor_result["responses"]}


    async def update_messages(
        self,
        sprite_id: str,
        messages: list[BaseMessage],
        skip_hooks: bool = False,
    ) -> None:
        """外部更新`messages`的唯一方式，避免了在图运行时无法修改messages的问题"""
        if not sprite_id:
            return
        config = {"configurable": {"thread_id": sprite_id}}
        state = await self.graph.aget_state(config)
        time_settings = store_manager.get_settings(sprite_id).time_settings
        current_times = Times.from_time_settings(time_settings)
        default_meta = SpritesMsgMetaOptionalTimes(
            creation_times=current_times
        )
        for message in messages:
            default_meta.fill_to(message)

        if not skip_hooks:
            info = OnUpdateMessagesInfo(
                messages_ctrl=ChangeableField(current=messages)
            )
            for plugin_name, plugin in self.plugins_with_name.items():
                control = None
                try:
                    control = await plugin.on_update_messages(sprite_id, info)
                except Exception:
                    logger.exception(f"plugin {plugin.name} on_update_messages failed")
                if control:
                    info = info._update_from_control(control, plugin_name)
            messages = info.messages_ctrl.current

        if not messages:
            return

        while state.next and state.next[0] == 'final':
            await asyncio.sleep(0.1)
            state = await self.graph.aget_state(config)
        # 如果图在运行交给图来更新，又或者只是出于不可打断状态，这特指从on_call_sprite开始到实际调用图的这一间隙
        if state.next or not self.is_interruptable(sprite_id):
            self.sprite_messages_to_update.setdefault(sprite_id, []).extend(messages)
        else:
            current_input_message_ids = [m.id for m in state.values.get("input_messages", []) if m.id]
            updated_input_messages = [m for m in messages if m.id in current_input_message_ids]
            current_message_ids = [m.id for m in state.values.get("messages", []) if m.id] + current_input_message_ids
            new_messages = [m for m in messages if m.id not in current_message_ids]
            await self.graph.aupdate_state(config, {
                "messages": messages,
                "new_messages": new_messages,
                "input_messages": updated_input_messages
            }, as_node='final')

    async def get_messages(self, sprite_id: str) -> list[AnyMessage]:
        """外部获取`messages`的唯一方式，返回会包括使用`update_messages`但还没来得及更新的消息"""
        state = await self.graph.aget_state({"configurable": {"thread_id": sprite_id}})
        messages: list[AnyMessage] = state.values.get("messages", [])
        interrupt_messages = self._process_interrupt_data(sprite_id, self.sprite_interrupt_datas.get(sprite_id, {}))
        input_messages: list[AnyMessage] = state.values.get("input_messages", [])
        update_messages = self.sprite_messages_to_update.get(sprite_id, [])
        if interrupt_messages:
            messages = add_messages(messages, interrupt_messages)
        if input_messages:
            messages = add_messages(messages, input_messages)
        if update_messages:
            messages = add_messages(messages, update_messages)
        return messages

    def _pop_messages_to_update(
        self,
        sprite_id: str,
        sprite_run_id: str,
        current_messages: Optional[list[AnyMessage]] = None,
        current_new_messages: Optional[list[AnyMessage]] = None
    ) -> tuple[list[BaseMessage], list[BaseMessage]]:
        """仅限图节点使用。用于在图运行时获取`sprite_messages_to_update`中可能存在的消息

        返回一个元组，第一个元素是需要更新至`messages`的消息列表，第二个元素是需要更新至`new_messages`的消息列表

        如果未提供`current_messages`和`current_new_messages`参数，则第二个元素返回空列表"""
        # 首先验证sprite运行ID是否能对应上，对不上意味着已开启新的运行，取消pop
        if self.is_current_run(sprite_id, sprite_run_id):
            update_messages = self.sprite_messages_to_update.pop(sprite_id, [])
            update_new_messages: list[BaseMessage] = []
            # 如果提供了current_messages和current_new_messages
            if update_messages and current_messages is not None and current_new_messages is not None:
                current_message_ids = [m.id for m in current_messages if m.id]
                current_new_message_ids = [m.id for m in current_new_messages if m.id]
                for m in update_messages:
                    if (
                        # 如果已经在new_messages里了，允许更新
                        m.id in current_new_message_ids or
                        # 如果不在messages里，说明确实是新消息
                        m.id not in current_message_ids
                        # 否则，这就只是在更新旧消息，而非新增消息
                    ):
                        update_new_messages.append(m)
            return update_messages, update_new_messages
        return [], []


    async def wait_until_interruptable(self, sprite_id: str) -> None:
        """等待直到sprite可被打断"""
        if self.sprite_interruptable.get(sprite_id) is not None:
            await self.sprite_interruptable[sprite_id].wait()

    def is_interruptable(self, sprite_id: str) -> bool:
        """检查sprite是否可被打断"""
        if self.sprite_interruptable.get(sprite_id) is not None:
            return self.sprite_interruptable[sprite_id].is_set()
        return True

    def set_is_interruptable(self, sprite_id: str) -> None:
        """设置sprite可被打断"""
        self.sprite_interruptable.setdefault(sprite_id, asyncio.Event()).set()

    def set_is_not_interruptable(self, sprite_id: str) -> None:
        """清除sprite可被打断状态"""
        self.sprite_interruptable.setdefault(sprite_id, asyncio.Event()).clear()


    @staticmethod
    def _process_interrupt_data(sprite_id: str, interrupt_data: InterruptData, current_times: Optional[Times] = None) -> list[BaseMessage]:
        """处理中断数据，将其转换为图运行时需要的格式"""
        interrupt_messages = []
        if chunk := interrupt_data.get('chunk'):
            # 将被中断的chunk加入messages（add_messages会自动处理）
            interrupt_messages.append(chunk)
            called_tool_messages = interrupt_data["called_tool_messages"]
            called_tool_messages_with_id = {tool_message.tool_call_id: tool_message for tool_message in called_tool_messages}
            called_tool_message_ids = called_tool_messages_with_id.keys()
            # 用可获取到的最晚的tool_message的创建时间作为新的工具消息的创建时间，chunk的创建时间则作为兜底
            if called_tool_messages:
                last_tool_message_metadata = SpritesMsgMeta.parse(called_tool_messages[-1])
                last_creation_times = last_tool_message_metadata.creation_times
            else:
                #last_creation_times = interrupt_data.get('last_chunk_times', current_times)
                last_creation_times = interrupt_data['last_chunk_times']
            # 对于chunk中出现的每个tool_call进行检查
            for tool_call in chunk.tool_calls:
                tool_call_id = tool_call.get('id')
                if tool_call_id:
                    # 如果called_tool_messages中已存在工具消息，则直接添加
                    if tool_call_id in called_tool_message_ids:
                        interrupt_messages.append(called_tool_messages_with_id[tool_call_id])
                    # 如果是send_message，被打断前的消息是依然存在的
                    elif tool_call["name"] == SEND_MESSAGE:
                        interrupt_messages.append(SpritesMsgMeta(
                            creation_times=last_creation_times,
                            message_type=DEFAULT_TOOL_MSG_TYPE,
                            is_action_only_tool=True
                        ).set_to(ToolMessage(
                            content=f'{SEND_MESSAGE_TOOL_CONTENT}（尽管当前调用被打断，被打断前的消息也已经发送成功）',
                            name=SEND_MESSAGE,
                            tool_call_id=tool_call_id,
                        )))
                    # 对于其他tool_call，则直接添加取消执行消息
                    else:
                        interrupt_messages.append(SpritesMsgMeta(
                            creation_times=last_creation_times,
                            message_type=DEFAULT_TOOL_MSG_TYPE
                        ).set_to(ToolMessage(
                            content='因当前调用被打断，此工具取消执行。',
                            name=tool_call["name"],
                            tool_call_id=tool_call_id,
                        )))

            if current_times is None:
                time_settings = store_manager.get_settings(sprite_id).time_settings
                current_times = Times.from_time_settings(time_settings)
            interrupt_messages.append(construct_system_message(
                content=f'''由于在你刚才输出时出现了“双重短信”（Double-texting，一般是由于用户在你输出期间又发送了一条或多条新的消息）的情况，你刚才的输出已被终止并截断，包括工具调用。
也因此你可能会发现自己刚才的输出并不完整且部分工具调用没有正确执行，这是正常现象。请根据接下来的新的消息重新考虑要输出的内容，或是否要重新调用刚才未完成的工具执行。
注意，工具`{SEND_MESSAGE}`是一个例外：由于它是实时流式输出的，不用等到工具调用的参数全部输出才执行，所以就算被“双发”截断了工具调用，用户也能看到已经输出的部分。这就相当于是你说话被打断了。''',
                times=current_times
            ))

        return interrupt_messages


    def is_sprite_running(self, sprite_id: str) -> bool:
        return bool(self.sprite_run_ids.get(sprite_id))

    def get_current_run_id(self, sprite_id: str) -> Optional[str]:
        return self.sprite_run_ids.get(sprite_id)

    def is_current_run(self, sprite_id: str, sprite_run_id: str) -> bool:
        return self.sprite_run_ids.get(sprite_id) == sprite_run_id

    def set_current_run(self, sprite_id: str, sprite_run_id: str) -> None:
        if last_run_id := self.sprite_run_ids.get(sprite_id):
            self.sprite_last_run_ids[sprite_id] = last_run_id
        self.sprite_run_ids[sprite_id] = sprite_run_id
        if sprite_id not in self.sprite_run_events:
            self.sprite_run_events[sprite_id] = asyncio.Event()
        else:
            self.sprite_run_events[sprite_id].clear()

    def clear_current_run(self, sprite_id: str) -> None:
        run_id = self.sprite_run_ids.pop(sprite_id, None)
        if run_id:
            self.sprite_last_run_ids[sprite_id] = run_id
        if sprite_id in self.sprite_run_events:
            self.sprite_run_events[sprite_id].set()

    async def wait_for_sprite_run(self, sprite_id: str) -> asyncio.Task:
        if sprite_id in self.sprite_run_events:
            await self.sprite_run_events[sprite_id].wait()


    def set_call_sprite_merging(self, sprite_id: str, sprite_run_id: str) -> None:
        if sprite_id in self.sprite_merging:
            self.sprite_merging[sprite_id][1] = sprite_run_id
            self.sprite_merging[sprite_id][0].clear()
        else:
            self.sprite_merging[sprite_id] = [asyncio.Event(), sprite_run_id]

    def set_call_sprite_merging_done(self, sprite_id: str, sprite_run_id: str) -> None:
        if self.sprite_merging.get(sprite_id) and self.sprite_merging[sprite_id][1] == sprite_run_id:
            self.sprite_merging[sprite_id][0].set()

    def is_call_sprite_merging(self, sprite_id: str) -> bool:
        if self.sprite_merging.get(sprite_id):
            return not self.sprite_merging[sprite_id][0].is_set()
        else:
            return False

    async def wait_for_call_sprite_merging(self, sprite_id: str) -> None:
        if sprite_id in self.sprite_merging:
            await self.sprite_merging[sprite_id][0].wait()

    def get_merging_call_sprite_run_id(self, sprite_id: str) -> Optional[str]:
        if sprite_id in self.sprite_merging:
            return self.sprite_merging[sprite_id][1]
        else:
            return None



def parse_agent_state(agent_state: list[StateEntry]) -> str:
    return '- ' + '\n- '.join([string for string in agent_state])
