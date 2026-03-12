from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
from typing import Any, Annotated
from uuid import uuid4
import random
from langchain_core.messages import ToolMessage, AIMessage
from langchain.tools import ToolRuntime, tool

from become_human.times import format_time, Times
from become_human.store.manager import store_manager
from become_human.types.manager import CallSpriteRequest
from become_human.types.main import MainState

from become_human.plugins.memory.base import memory_manager, InitialMemory, format_retrieved_memory_groups
from become_human.plugins.memory.message import MemoryMsgMeta, get_all_retrieved_memory_ids
from become_human.plugins.memory.store import MemoryConfig
from become_human.plugins.memory.recycling import connect_last_memory


RETRIEVE_MEMORIES = 'retrieve_memories'

backup_schema = {
    '$defs': {
        'time': {
            'type': 'object',
            'description': '''creation_time_range中的描述时间点的结构体。所有参数都是可选的，每个参数都相当于是在最终的filter中添加了一个参数，所以可以自由组合。若全为空，则表示无时间限制。
比如，可以通过只填写hour参数来表示上下午等时间段而不限定是哪一天的上下午。''',
            'properties': {
                'year': {
                    'anyOf': [{'maximum': 9999, 'minimum': 1, 'type': 'integer'}, {'type': 'null'}],
                    'description': '年（1~9999）',
                    'default': None,
                },
                'month': {
                    'anyOf': [{'maximum': 12, 'minimum': 1, 'type': 'integer'}, {'type': 'null'}],
                    'description': '月（1~12）',
                    'default': None,
                },
                'day': {
                    'anyOf': [{'maximum': 31, 'minimum': 1, 'type': 'integer'}, {'type': 'null'}],
                    'description': '日（1~31）',
                    'default': None,
                },
                'hour': {
                    'anyOf': [{'maximum': 23, 'minimum': 0, 'type': 'integer'}, {'type': 'null'}],
                    'description': '小时（0~23）',
                    'default': None,
                },
                'minute': {
                    'anyOf': [{'maximum': 59, 'minimum': 0, 'type': 'integer'}, {'type': 'null'}],
                    'description': '分钟（0~59）',
                    'default': None,
                },
                'second': {
                    'anyOf': [{'maximum': 59, 'minimum': 0, 'type': 'integer'}, {'type': 'null'}],
                    'description': '秒钟（0~59）',
                    'default': None,
                },
                'weekday': {
                    'anyOf': [{'maximum': 7, 'minimum': 1, 'type': 'integer'}, {'type': 'null'}],
                    'description': '星期几（1~7）',
                    'default': None,
                },
                'week': {
                    'anyOf': [{'maximum': 53, 'minimum': 1, 'type': 'integer'}, {'type': 'null'}],
                    'description': '第几周（1~53）',
                    'default': None,
                },
            }
        }
    },
    'properties': {
        'creation_time_range': {
            'type': 'object',
            'description': '''用于限定检索结果的创建时间范围，默认为空，表示不作限制。
由于是使用filter直接在程序上过滤掉了这个范围以外的记忆，所以比起在search_string中描述时间范围（只有episodic类型的记忆适合在search_string中描述时间），使用creation_time_range会更快且更准确（但不意味着使用了creation_time_range就不能在search_string中描述时间范围了）。
但也需要注意这里的create_time，准确来说指的是**记忆的创建时间**，这个时间是在记忆被创建时自动生成的。
意味着可能会出现“张三上星期感冒了。”这条记忆在张三实际感冒后一星期才创建，因为你是在一星期后才知道的这件事。
此时如果使用creation_time_range来限定到张三感冒实际发生时的时间点，反而会检索不到“张三上星期感冒了。”这条记忆。
总之不建议在较小范围使用creation_time_range，特别是当检索他人信息时，可能会出现这种意外情况。''',
            'properties': {
                'start': {
                    '$ref': '#/$defs/time',
                    'description': '起始时间点，也可在没有结束时间点的情况下单独使用，意为只限制起始时间不限制结束时间。',
                },
                'end': {
                    '$ref': '#/$defs/time',
                    'description': '结束时间点，也可在没有起始时间点的情况下单独使用，意为只限制结束时间不限制起始时间。',
                }
            }
        }
    }
}

RM_MEMORY_TYPE_PROMPTS = {
    "original": '''## 对于original类型的记忆
original类型的记忆是直接按对话轮次原文存储的，对于其内容来说没有任何规律，但同时也会保留其固定格式，其中包含：
- 发起动作的人名，或“我”。
- 时间信息，以完整的[%Y-%m-%d %H:%M:%S Week%W %A]格式，如[2025-11-30 09:30:00 Week47 Sunday]。
- 你自己在当时完整的内心想法。
- 你执行过的工具调用中包含的工具名称和参数，以及工具的返回结果。
这也意味着original类型记忆可能在大多数情况下不是特别适用，因为其一般包含大量无用信息，不如其他记忆类型精炼。
不过如果对于summary类型记忆（若有）提供的信息不够满意，想回忆更完整细节，这也是唯一的方式。
可以通过以上固定格式中包含的信息来检索，或直接搜索具体内容。
在面对下列示例问题时，适合选择着重检索original类型记忆：
- 想要根据现有信息知道回忆起张三当时具体说了什么（由于是向量检索所以时间只需尽可能精确）：
  {
    "search_string": "[2024-07-02 中午]张三：我想吃炸鸡。",
    "memory_type": "original"
  }
- 想要根据自己做过的事回忆起自己当时为什么会对东京塔有多高感兴趣：
  {
    "search_string": "我的动作：web_search({"query": "东京塔有多高？"})",
    "memory_type": "original"
  }''',
    "episodic": '''## 对于episodic类型的记忆
episodic类型的记忆尽可能包含了在你的上下文中发生的所有事件（的细节）。
episodic类型的所有记忆里都会包含尽可能准确到秒、第几周和星期几的绝对时间信息，具体为%Y-%m-%d %H:%M:%S Week%W %A格式，如2025-11-30 09:30:00 Week47 Sunday。
这意味着对于episodic类型的记忆来说，你可以在search_string中也像这样直接描述时间（因为是向量检索，不必完全精确），来找到指定时间可能发生的事情（如果还记得的话）。
在面对以下示例问题时，适合选择着重检索episodic类型记忆：
- 想知道，今年情人节我跟谁聊天了：
  {
    "search_string": "2025-02-14我和谁聊天了？",
    "memory_type": "episodic"
  }''',
    "semantic": '''## 对于semantic类型的记忆
semantic类型的记忆是从过往经历（包括可能的心理活动）中思考并得出的结论或猜测，形成的原子级语义信息。
如果你想知道什么是什么，什么怎么样诸如此类的信息，而非发生过什么，选择semantic。
在面对以下示例问题时，适合指定检索semantic类型记忆：
- 想知道，李四喜欢吃什么：
  {
    "search_string": "李四喜欢吃什么？",
    "memory_type": "semantic"
  }''',
    "summary": '''## 对于summary类型的记忆
summary类型的记忆是对过往经历的按时间范围区分的总结陈述，时间范围包含年、季、月、周、日四种。
在想要获取某时间范围内的更整体的信息总结时，这会很有用。
需注意，在指定检索summary类型记忆时，并不是在进行向量检索，而是根据你在search_string中提供的时间信息直接进行查询。
具体来说，时间信息应遵循格式（类似于ISO 8601）：<Year>[-Season|-Month][-Week][-Day]，以下是一些常用例子：
- 查询某年：2025、2021
- 查询某季：2025-Winter、2024-Autumn
- 查询某月：2025-12、2024-08
- 查询某周：2025-W44、2024-W25
- 查询某月某日：2025-11-30、2024-08-25
其他的一些更复杂的例子：
- 查询某周某日：2025-W44-01、2024-W25-07
- 查询某季某周：2025-Spring-W5、2024-Summer-W12[-03]（可以继续指定星期几）
- 查询某月某周：2025-11-W1、2024-08-W3[-07]（可以继续指定星期几）

提示，这里的季度划分具体来说为：
- 春季：3月、4月、5月
- 夏季：6月、7月、8月
- 秋季：9月、10月、11月
- 冬季：12月、1月、2月

在面对以下示例问题时，适合指定检索summary类型记忆：
- 想知道，我在今年（假设2025）十一月都做了什么：
  {
    "search_string": "2025-11",
    "memory_type": "summary"
  }''',
}
RM_MEMORY_TYPE_SUGGESTIONS = {
    "episodic": "- 需要回忆过往发生过的事情时，指定检索episodic类型的记忆。",
    "semantic": "- 需要查询从经历中思考得出的语义信息而非经历本身时，指定检索semantic类型的记忆。",
    "summary": "- 需要获取按时间范围区分的经历的总结时，指定检索summary类型的记忆。",
}
def construct_retrieve_memories_schema(sprite_id: str) -> dict[str, Any]:
    """根据激活的记忆类型构造检索记忆的schema"""
    current_memory_types = store_manager.get_model(sprite_id, MemoryConfig).memory_types
    rm_memory_types = []
    if 'original' in current_memory_types or 'summary' in current_memory_types:
        rm_memory_types.append('episodic')
    if 'reflective' in current_memory_types:
        rm_memory_types.append('semantic')
    if 'summary' in current_memory_types:
        rm_memory_types.append('summary')
    states = store_manager.get_states(sprite_id)
    settings = store_manager.get_settings(sprite_id)
    rm_schema = {
        # langchain在解析schema时会更倾向于使用工具的name而非schema中的title作为工具名
        # 但对于某些情况，如果只提供schema而不是工具时，意味着此时无法获取工具的name
        # 在这种情况下，schema顶层的title依然是有意义的
        # 不过对于其他各个属性的title，langchain都会将其删除，所以没有必要填写（除了当title作为属性名时）
        "title": RETRIEVE_MEMORIES,
        "description": f'''从你自己的向量数据库中检索信息。
你的向量数据库有一套自动从过往上下文中分析并存储信息的机制，而这个工具使你可以主动从这个数据库中检索信息。
对于角色扮演来说，如果你所扮演的是类人（非机器人之类）的角色，可以将数据库理解为大脑，检索理解为回忆，信息理解为记忆。

# 何时使用此工具？
- 不满足于被动检索提供的信息时。
- 需要获取更多私人的信息，而非互联网可以查询到的信息时。
- 当前场景需要你像人一样去主动回忆时，根据你所扮演的角色。
{'\n'.join([RM_MEMORY_TYPE_SUGGESTIONS.get(t, '') for t in rm_memory_types if RM_MEMORY_TYPE_SUGGESTIONS.get(t)])}

# 可检索的记忆类型
有{len(rm_memory_types)}种类型记忆可供检索，分别是：{'，'.join(rm_memory_types)}。以下是{len(rm_memory_types)}种不同类型记忆的说明，你可以根据这些信息来决定自己要着重检索哪种记忆类型。

{'\n\n'.join([RM_MEMORY_TYPE_PROMPTS.get(t, '') for t in rm_memory_types if RM_MEMORY_TYPE_PROMPTS.get(t)])}

# 可检索到的最早时间
由于记忆是在运行时收集存储的，所以显而易见的，你一般不会拥有早于第一次初始化时间之前的记忆。
该时间为 {format_time(Times.from_time_settings(settings.time_settings, states.born_at).sprite_world_datetime)}。

# 重复检索
这是一项额外功能，如果你使用了在之前的上下文中使用过的相同的该工具的工具调用参数，将触发重复检索。
这有什么用？
默认情况下，检索结果会自动剔除掉当前上下文已存在的检索到的记忆，避免被重复信息挤占上下文空间。
而如果触发了重复检索，则不会剔除掉之前使用该工具调用参数时检索到的记忆。
如果你因一些原因如遇到了「模糊的记忆」并想要重复检索以获得更完整的信息时，请尝试使用重复检索。''',
        "type": "object",
        "properties": {
            "search_string": {
                'type': 'string',
                'description': '''用于检索对应记忆类型的查询字符串，一般来说应使用语义准确的疑问句（除非是summary类型），具体参考各记忆类型给出的示例。
除"我"以外，不要使用代词，请使用全称。举例：不应使用"他"这种人称代词，应为某人的全名如"李四"；但对于"我"来说，请保持"我"。'''
            },
            "memory_type": {
                'type': 'string',
                'default': '',
                'description': f'''要指定检索的记忆类型。默认为空，表示会混合检索各种记忆。
有{len(rm_memory_types)}种类型记忆可供选择，分别是：{'，'.join(rm_memory_types)}。''',
                'enum': [''] + rm_memory_types
            }
        },
        "required": ["search_string"]
    }
    return rm_schema
@tool(RETRIEVE_MEMORIES, response_format="content_and_artifact")
async def retrieve_memories_tool(
    runtime: ToolRuntime[CallSpriteRequest, MainState],
    search_string: str,
    memory_type: str = '',
    #creation_time_range_start: str = '',
    #creation_time_range_end: str = ''
) -> tuple[str, MemoryMsgMeta]:
    """placeholder"""
    memory_types = store_manager.get_model(runtime.context.sprite_id, MemoryConfig).memory_types
    if not memory_types:
        raise ValueError("没有可检索的记忆类型，可能是配置错误，请不要继续使用此工具。")
    creation_time_range_start, creation_time_range_end = None, None # 目前取消了这个参数
    if creation_time_range_start:
        try:
            start_time = datetime.strptime(creation_time_range_start.strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError(f'无法解析输入中的 creation_time_range_start！请重新检查你的输入是否符合"2021-01-01 00:00:00"这样的格式！(也即%Y-%m-%d %H:%M:%S)。若不需要此功能请留空。')
    if creation_time_range_end:
        try:
            end_time = datetime.strptime(creation_time_range_end.strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError(f'无法解析输入中的 creation_time_range_end！请重新检查你的输入是否符合"2023-01-01 23:59:59"这样的格式！(也即%Y-%m-%d %H:%M:%S)。若不需要此功能请留空。')

    if memory_type == 'summary':
        time_settings = store_manager.get_settings(sprite_id).time_settings
        try:
            time_parts = search_string.split('-')
            time_kwargs = {}
            time_granularity = None
            for i, p in enumerate(time_parts):
                if i == 0:
                    time_kwargs['year'] = int(p)
                    time_granularity = 'year'
                elif p.lower() in ('spring', 'summer', 'autumn', 'fall', 'winter'):
                    time_kwargs['month'] = {'spring': 3, 'summer': 6, 'autumn': 9, 'fall': 9, 'winter': 12}[p.lower()]
                    time_granularity = 'season'
                elif p.upper().startswith('W'):
                    time_kwargs['weeks'] = int(p[1:])
                    time_granularity = 'week'
                elif time_granularity == 'week':
                    time_kwargs.pop('day', None)
                    time_kwargs['weekday'] = int(p) - 1
                    time_granularity = 'day'
                elif time_granularity == 'month':
                    time_kwargs.pop('weekday', None)
                    time_kwargs['day'] = int(p)
                    time_granularity = 'day'
                else:
                    time_kwargs['month'] = int(p)
                    time_granularity = 'month'
            summary_datetime = datetime(
                time_kwargs.pop('year'),
                time_kwargs.pop('month', 1),
                time_kwargs.pop('day', 1),
                tzinfo=time_settings.time_zone.tz()
            )
            if time_kwargs:
                # 如果有周相关的参数，先回到周1再计算
                # 这也使得可能会回到上一年/上个月的最后一周
                # 就是这么设计的
                summary_datetime -= timedelta(days=summary_datetime.weekday())
                summary_datetime += relativedelta(**time_kwargs)
            if not time_granularity:
                raise ValueError(f'无法解析输入中的时间字符串 {search_string}！请重新检查你的输入是否符合格式！')
        except Exception:
            raise ValueError(f'无法解析输入中的时间字符串 {search_string}！请重新检查你的输入是否符合格式！')
        summary_datetime = summary_datetime.astimezone(timezone.utc)
        summary = await memory_manager.get_memories(
            sprite_id,
            memory_type,
            where={
                '$and': [
                    {'summary_time_granularity': time_granularity},
                    {'summary_date_iso': summary_datetime.date().isoformat()},
                ]
            }
        )
        if not summary:
            return '没有找到该时间点的记忆总结，也许根本就不存在，又或许已经遗忘了。', None
        summary = summary[0]
        return '主动记忆检索结果：\n\n' + summary.page_content, MemoryMsgMeta(
            do_not_store=True,
            retrieved_memory_ids=[summary.id],
        )


    sprite_id = runtime.context.sprite_id
    messages = runtime.state.messages

    store_settings = store_manager.get_settings(sprite_id)
    retrieval_configs = store_manager.get_model(sprite_id, MemoryConfig)
    try:
        retrieval_configs = getattr(retrieval_configs, f'active_{memory_type}_retrieval_configs') if memory_type else retrieval_configs.active_common_retrieval_configs
    except AttributeError:
        raise ValueError(f'无法找到指定的记忆类型 {memory_type} 的检索配置！请检查你输入的参数是否正确。')


    # 剔除已检索过的记忆
    message_ids = [m.id for m in messages if m.id]
    retrieved_memory_ids = get_all_retrieved_memory_ids(messages)
    exclude_memory_ids = list(set(message_ids + retrieved_memory_ids))

    # 重复检索逻辑，若重复检索则不剔除检索过的那些记忆
    repeated_tool_call_ids = []
    repeat_content = ''
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for tool_call in m.tool_calls:
            if (
                tool_call.name == RETRIEVE_MEMORIES and
                tool_call.args.get("search_string") == search_string and
                tool_call.args.get("memory_type") == memory_type and
                tool_call.id
            ):
                repeated_tool_call_ids.append(tool_call.id)
                repeat_content = '由于你使用了之前上下文中使用过的该工具的调用参数，触发了重复检索，该次检索不会剔除掉之前使用该工具调用参数时检索到的记忆。\n\n'
    if repeated_tool_call_ids:
        include_memory_ids = set(get_all_retrieved_memory_ids([
            m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id in repeated_tool_call_ids
        ]))
        if include_memory_ids:
            exclude_memory_ids = [id for id in exclude_memory_ids if id not in include_memory_ids]
        else:
            repeat_content = '由于你使用了之前上下文中使用过的该工具的调用参数，触发了重复检索，但是没有找到之前检索到的记忆，该次检索将依然剔除任何已被召回到当前上下文的记忆。这可能是由于当时的检索出错了，又或者是其他未知错误。\n\n'

    groups = await memory_manager.retrieve_memories(
        sprite_id=sprite_id,
        retrieval_configs=retrieval_configs,
        #memory_type=memory_type if memory_type else None,
        search_string=search_string,
        creation_time_range_start=start_time,
        creation_time_range_end=end_time,
        exclude_memory_ids=exclude_memory_ids
    )
    content = repeat_content + format_retrieved_memory_groups(groups, store_settings.time_settings.time_zone)
    artifact = MemoryMsgMeta(
        do_not_store=True,
        retrieved_memory_ids=[group.source_memory.doc.id for group in groups]
    )
    return content, artifact

@tool('add_memory')
async def add_memory_tool(
    runtime: ToolRuntime[CallSpriteRequest],
    content: Annotated[str, "记忆内容"],
) -> str:
    """主动添加一条记忆，将作为semantic类型的记忆被记住（一段时间）"""
    if not content.strip():
        return "记忆内容不能为空"
    sprite_id = runtime.context.sprite_id
    base_ttl = store_manager.get_model(sprite_id, MemoryConfig).memory_base_ttl
    new = str(uuid4())
    last = await connect_last_memory(sprite_id, 'reflective', [new])
    await memory_manager.add_memories([
        InitialMemory(
            content=content,
            type='reflective',
            creation_times=Times.from_time_settings(store_manager.get_settings(sprite_id).time_settings),
            ttl=int(random.expovariate(0.4) * base_ttl),
            id=new,
            previous_memory_id=last
        ),
        sprite_id
    ])
    return "记忆添加成功"
