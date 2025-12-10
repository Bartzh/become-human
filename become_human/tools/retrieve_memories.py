from typing import Annotated, Optional, Any
from datetime import datetime
from langchain_core.messages import AnyMessage
from langchain.tools import InjectedState, tool
from langchain_core.runnables import RunnableConfig
from langchain.messages import HumanMessage, ToolMessage, AIMessage

from become_human.store_manager import store_manager
from become_human.memory import get_activated_memory_types, memory_manager, parse_retrieved_memory_groups
from become_human.types_main import get_retrieved_memory_ids

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
- 时间信息，以完整的[%Y-%m-%d %H:%M:%S %A]格式，如[2025-11-30 09:30:00 Sunday]。
- 你自己在当时完整的内心想法。
- 你执行过的工具调用中包含的工具名称和参数，以及工具的返回结果。
这也意味着orginal类型记忆可能在大多数情况下不是特别适用，因为其一般包含大量无用信息，不如其他记忆类型精炼。
不过如果对于episodic类型记忆（若有）提供的信息不够满意，想回忆更完整细节，这也是唯一的方式。
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
episodic类型的记忆是对于在你的上下文中发生的所有独立的小的事件的一句话总结陈述，所有记忆均为仅表达一个事实的原子级信息。在想要知晓你的过往发生过什么时，这会很有用。但请注意，episodic类型记忆中一般不会包含你自己的内心想法。
episocid类型的记忆还有两个特点，一是几乎所有记忆都会包含“我”关键字，因为显而易见的原因：episodic类型记忆记录的就是“我”（也就是你）的经历。
二是episodic类型的所有记忆里都会包含尽可能准确到秒和周几的绝对时间信息，具体为%Y-%m-%d %H:%M:%S %A格式，如2025-11-30 09:30:00 Sunday。
这意味着对于episodic类型的记忆来说，你可以在search_string中也像这样直接描述时间，来找到指定时间可能存在的记忆。
在面对以下示例问题时，适合选择着重检索episodic类型记忆：
- 想知道，今年情人节我跟谁聊天了：
  {
    "search_string": "2025-02-14我和谁聊天了？",
    "memory_type": "episodic"
  }
- 想知道，我在今年十一月都做了什么：
  {
    "search_string": "我在2025-11都做了些什么？",
    "memory_type": "episodic"
  }''',
    "reflective": '''## 对于reflective类型的记忆
reflective类型的记忆是从过往经历（包括你自己的内心想法）中思考并得出的结论或猜测，形成的原子级语义信息。
如果你想知道什么是什么，什么怎么样诸如此类的信息，而非发生过什么，选择reflective。
若配合使用creation_time_range，则意为：查询某段时间所产生的（某某相关的）感悟。
在面对以下示例问题时，适合选择着重检索reflective类型记忆：
- 想知道，张三毕业于哪所大学：
  {
    "search_string": "张三曾毕业于哪所大学？",
    "memory_type": "reflective"
  }
- 想知道，李四喜欢吃什么：
  {
    "search_string": "李四喜欢吃什么？",
    "memory_type": "reflective"
  }
- 想知道，阿哲的工作状况：
  {
    "search_string": "阿哲的工作状况如何？",
    "memory_type": "reflective"
  }''',
}
RM_MEMORY_TYPE_SUGGESTIONS = {
    "original": "- 需要回忆过往经历的完整细节时，着重检索original类型的记忆。",
    "episodic": "- 需要回忆过往发生过的事情时，着重检索episodic类型的记忆。",
    "reflective": "- 需要查询从经历中思考得出的语义信息而非经历本身时，着重检索reflective类型的记忆。",
}
rm_memory_types = get_activated_memory_types()
rm_schema = {
    #"title": "RetrieveMemoriesInputs",
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
            'description': '''用于检索对应记忆类型的查询字符串，一般来说应使用语义准确的疑问句，具体参考各记忆类型给出的示例。
除"我"以外，不要使用代词，请使用全称。举例：不应使用"他"这种人称代词，应为某人的全名如"李四"；但对于"我"来说，请保持"我"。'''
        },
        "memory_type": {
            'type': 'string',
            'default': '',
            'description': f'''要着重检索的记忆类型。默认为空，表示所有记忆类型使用相同的权重进行检索。
需要提醒的是，无论你选择检索哪种类型的记忆，其他类型的记忆仍会出现，你的选择只是提高了该记忆类型出现的比例而已。尽管如此，这仍然重要。
有{len(rm_memory_types)}种类型记忆可供选择，分别是：{'，'.join(rm_memory_types)}。''',
            'enum': [''] + rm_memory_types
        }
    },
    "required": ["search_string"]
}
@tool(response_format="content_and_artifact", args_schema=rm_schema)
async def retrieve_memories(
    messages: Annotated[list[AnyMessage], InjectedState('messages')],
    config: RunnableConfig,
    search_string: str,
    memory_type: str = '',
    #creation_time_range_start: str = '',
    #creation_time_range_end: str = ''
) -> tuple[str, dict[str, Any]]:
    memory_types = get_activated_memory_types()
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
    thread_id = config["configurable"]["thread_id"]

    # 本次循环中调用此工具次数越多，强度越高
    invoke_count = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        elif isinstance(m, ToolMessage) and m.name == "retrieve_memories":
            invoke_count += 1
    strength = 1 + min(invoke_count * 0.5, 1) # 目前主动检索的强度固定初始为1
    store_settings = await store_manager.get_settings(thread_id)
    retrieval_config = store_settings.retrieval.active_retrieval_config.model_copy(update={"strength": strength})

    # 剔除已检索过的记忆
    message_ids = [m.id for m in messages if m.id]
    retrieved_memory_ids = get_retrieved_memory_ids(messages)
    exclude_memory_ids = list(set(message_ids + retrieved_memory_ids))

    # 重复检索逻辑，若重复检索则不剔除检索过的那些记忆
    repeated_tool_call_ids = []
    repeat_content = ''
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for tool_call in m.tool_calls:
            if (
                tool_call.name == "retrieve_memories" and
                tool_call.args.get("search_string") == search_string and
                tool_call.args.get("memory_type") == memory_type and
                tool_call.id
            ):
                repeated_tool_call_ids.append(tool_call.id)
                repeat_content = '由于你使用了之前上下文中使用过的该工具的调用参数，触发了重复检索，该次检索不会剔除掉之前使用该工具调用参数时检索到的记忆。\n\n'
    if repeated_tool_call_ids:
        include_memory_ids = []
        for m in messages:
            if not isinstance(m, ToolMessage):
                continue
            if m.tool_call_id in repeated_tool_call_ids and isinstance(m.artifact, dict):
                include_memory_ids.extend(m.artifact.get("bh_retrieved_memory_ids", []))
        include_memory_ids = set(include_memory_ids)
        if include_memory_ids:
            exclude_memory_ids = [id for id in exclude_memory_ids if id not in include_memory_ids]
        else:
            repeat_content = '由于你使用了之前上下文中使用过的该工具的调用参数，触发了重复检索，但是没有找到之前检索到的记忆，该次检索将依然剔除任何已被召回到当前上下文的记忆。这可能是由于当时的检索出错了，又或者是其他未知错误。\n\n'

    groups = await memory_manager.retrieve_memories(
        thread_id=thread_id,
        retrieval_config=retrieval_config,
        memory_type=memory_type if memory_type else None,
        search_string=search_string,
        creation_time_range_start=start_time,
        creation_time_range_end=end_time,
        exclude_memory_ids=exclude_memory_ids
    )
    content = repeat_content + parse_retrieved_memory_groups(groups, store_settings.main.time_settings)
    artifact = {"bh_do_not_store": True, "bh_retrieved_memory_ids": [group.source_memory.doc.id for group in groups]}
    return content, artifact
