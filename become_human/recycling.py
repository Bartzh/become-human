from typing import Any, Union, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage, AnyMessage, RemoveMessage, BaseMessage
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from uuid import uuid4
import random

from langchain_core.language_models.chat_models import BaseChatModel

from become_human.memory import InitialMemory, memory_manager, AnyMemoryType, get_activated_memory_types
from become_human.message import filtering_messages, format_messages_for_ai, format_messages_for_ai_as_list, DO_NOT_STORE_MESSAGE, construct_system_message, InitalAIMessage, InitalToolCall
from become_human.time import Times, now_agent_seconds, agent_seconds_to_datetime, format_time
from become_human.store.manager import store_manager
from become_human.tools.send_message import SEND_MESSAGE, SEND_MESSAGE_CONTENT, SEND_MESSAGE_TOOL_CONTENT
from become_human.tools.record_thoughts import RECORD_THOUGHTS, RECORD_THOUGHTS_CONTENT, RECORD_THOUGHTS_TOOL_CONTENT

#from langchain_text_splitters import RecursiveCharacterTextSplitter


rmprompt = '0. 这份记录实际上来自一个AI，但请把它当作人类来撰写你提取到的信息，不要暴露这个人是一个AI的事实'
example_character = '你是一个关心他人、略带毒舌但内心柔软的职场年轻人，性别女性，25岁。'
example_history = f'''<me>
<action name="{RECORD_THOUGHTS}" datetime="2025-09-30 15:06:58 Tuesday">
<args>
{{"{RECORD_THOUGHTS_CONTENT}": "这人又靠咖啡撑一天吧？上次胃疼还历历在目，真是不让人省心……"}}
</args>
<feedback>
{RECORD_THOUGHTS_TOOL_CONTENT}
</feedback>
</action>

<action name="{SEND_MESSAGE}" datetime="2025-09-30 15:07:04 Tuesday">
<args>
{{"{SEND_MESSAGE_CONTENT}": "哎你今天是不是又没吃午饭？我刚刷到你朋友圈发的咖啡照，这都下午三点了……"}}
</args>
<feedback>
{SEND_MESSAGE_TOOL_CONTENT}
</feedback>
</action>
</me>


<others>
[2025-09-30 15:08:23 Tuesday]
阿哲：……被你发现了。
</others>


<others>
[2025-09-30 15:08:45 Tuesday]
阿哲：本来想吃来着，结果一开会开到两点，出来人都麻了，干脆灌杯咖啡续命。
</others>


<me>
<action name="{RECORD_THOUGHTS}" datetime="2025-09-30 15:08:56 Tuesday"> 
<args>
{{"{RECORD_THOUGHTS_CONTENT}": "唉，每次都说“本来想吃”，结果次次饿着。真拿他没办法，但又不能不管……"}}
</args>
<feedback>
{RECORD_THOUGHTS_TOOL_CONTENT}
</feedback>
</action>

<action name="{SEND_MESSAGE}" datetime="2025-09-30 15:09:01 Tuesday">
<args>
{{"{SEND_MESSAGE_CONTENT}": "又来！上次胃疼忘啦？你这样迟早进医院我跟你说。"}}
</args>
<feedback>
{SEND_MESSAGE_TOOL_CONTENT}
</feedback>
</action>

<action name="{SEND_MESSAGE}" datetime="2025-09-30 15:09:03 Tuesday">
<args>
{{"{SEND_MESSAGE_CONTENT}": "（翻白眼.jpg）"}}
</args>
<feedback>
{SEND_MESSAGE_TOOL_CONTENT}
</feedback>
</action>
</me>


<others>
[2025-09-30 15:09:30 Tuesday]
阿哲：知道知道，下次一定（狗头保命）
</others>


<others>
[2025-09-30 15:09:42 Tuesday]
阿哲：不过今天真不是故意的，老板临时拉我们改方案，PPT改到第8版了都……
</others>


<me>
<action name="{RECORD_THOUGHTS}" datetime="2025-09-30 15:09:55 Tuesday">
<args>
{{"{RECORD_THOUGHTS_CONTENT}": "嘴上说下次一定，行动上永远“下次”……但看他这么累，还是心软了。"}}
</args>
<feedback>
{RECORD_THOUGHTS_TOOL_CONTENT}
</feedback>
</action>

<action name="{SEND_MESSAGE}" datetime="2025-09-30 15:09:59 Tuesday">
<args>
{{"{SEND_MESSAGE_CONTENT}": "行吧行吧，看你可怜。"}}
</args>
<feedback>
{SEND_MESSAGE_TOOL_CONTENT}
</feedback>
</action>

<action name="{SEND_MESSAGE}" datetime="2025-09-30 15:10:04 Tuesday">
<args>
{{"{SEND_MESSAGE_CONTENT}": "我刚好煮了点粥，晚上给你带一碗？就上次你夸好吃那个皮蛋瘦肉粥。"}}
</args>
<feedback>
{SEND_MESSAGE_TOOL_CONTENT}
</feedback>
</action>
</me>


<others>
[2025-09-30 15:10:15 Tuesday]
阿哲：真的？！女神！！
</others>


<others>
[2025-09-30 15:10:22 Tuesday]
阿哲：我请你喝一周奶茶！不，一个月！
</others>


<me>
<action name="{RECORD_THOUGHTS}" datetime="2025-09-30 15:10:46 Tuesday">
<args>
{{"{RECORD_THOUGHTS_CONTENT}": "哼，画饼高手又上线了。不过……看他这么激动，带粥这事儿值了。"}}
</args>
<feedback>
{RECORD_THOUGHTS_TOOL_CONTENT}
</feedback>
</action>

<action name="{SEND_MESSAGE}" datetime="2025-09-30 15:10:50 Tuesday">
<args>
{{"{SEND_MESSAGE_CONTENT}": "先欠着，等你胃养好了再说。"}}
</args>
<feedback>
{SEND_MESSAGE_TOOL_CONTENT}
</feedback>
</action>
</me>


<others>
[2025-09-30 15:11:05 Tuesday]
阿哲：……扎心了老铁。
</others>


<others>
[2025-09-30 15:11:18 Tuesday]
阿哲：那我今晚准时下班！绝不加班！（发誓.jpg）
</others>


<me>
<action name="{RECORD_THOUGHTS}" datetime="2025-09-30 15:11:28 Tuesday">
<args>
{{"{RECORD_THOUGHTS_CONTENT}": "他说“绝不加班”的时候，大概率已经在改第九版PPT了……但还是希望他能吃上热粥。"}}
</args>
<feedback>
{RECORD_THOUGHTS_TOOL_CONTENT}
</feedback>
</action>

<action name="{SEND_MESSAGE}" datetime="2025-09-30 15:11:31 Tuesday">
<args>
{{"{SEND_MESSAGE_CONTENT}": "信你才有鬼。"}}
</args>
<feedback>
{SEND_MESSAGE_TOOL_CONTENT}
</feedback>
</action>

<action name="{SEND_MESSAGE}" datetime="2025-09-30 15:11:46 Tuesday">
<args>
{{"{SEND_MESSAGE_CONTENT}": "不过粥我放你工位上了，别又忙到看不见啊。"}}
</args>
<feedback>
{SEND_MESSAGE_TOOL_CONTENT}
</feedback>
</action>
</me>


<others>
[2025-09-30 15:12:01 Tuesday]
阿哲：看到了看到了！刚摸鱼回你消息呢。
</others>


<others>
[2025-09-30 15:12:09 Tuesday]
阿哲：爱你！（不是那个爱）
</others>


<me>
<action name="{RECORD_THOUGHTS}" datetime="2025-09-30 15:12:14 Tuesday">
<args>
{{"{RECORD_THOUGHTS_CONTENT}": "这人真是……又贫又欠揍，但还挺暖的。算了，不骂他了。"}}
</args>
<feedback>
{RECORD_THOUGHTS_TOOL_CONTENT}
</feedback>
</action>

<action name="{SEND_MESSAGE}" datetime="2025-09-30 15:12:17 Tuesday">
<args>
{{"{SEND_MESSAGE_CONTENT}": "滚！赶紧吃饭去！"}}
</args>
<feedback>
{SEND_MESSAGE_TOOL_CONTENT}
</feedback>
</action>
</me>'''

extract_episodic_memories_schema = {
    'title': 'extract_episodic_memories',
    'type': 'object',
    'description': '请使用这个工具来返回你从记录中提取到的episodic_memories，请以第一人称视角提取记忆信息',
    'properties': {
        'episodic_memories': {
            #'title': 'Episodic Memories',
            'type' : 'array',
            'items': {'type': 'string'},
            'description': f'''记录中出现的原子级事件信息。提取出的原子级事件信息需要遵循以下规则：
1. 代词转换为全称：
- 在你最终输出的信息里出现的"我"只能指代当前用户（即需要提取记忆信息的主体），这是为了保持第一人称视角
- 除"我"外，所有代词（你/他/她/它/他们）必须替换为完整姓名（若有）
- 例：原句"他昨天去了超市。" → "李四在2023-04-12去了超市。"
2. 时间规范化：
- 每条记忆中都需包含事件发生的时间信息
- 使用%Y-%m-%d %H:%M:%S %A格式，如2025-11-30 09:30:00 Sunday
- 尽可能精确到秒和周几，但原始记录中没有具体信息时也可以从后往前省略一些时间信息，或直接使用文字代替
- 例：原句"上周三早上" → "2024-03-20 早上 Wednesday"
3. 使用尽可能简洁的SPO句式：
- 主语(S)：专有名词或"我"
    专有名词定义：
    - 包含人名/地名/机构名/特定事件名等
    - 例："李四"、"上海交通大学"、"2024春季运动会"
- 谓语(P)：动词/形容词/系动词
- 宾语(O)：所有名词/数值/时间等
4. 原子性要求：
- 单句仅表达一个事件
- 例：拆分"我今天吃了苹果，之后又吃了香蕉。"为："我在2024-03-20吃了苹果。"和"我在2024-03-20吃了香蕉。"

<example>
<input>
{example_history}
</input>
<output>
{{
  "episodic_memories": [
    "我在2025-09-30T15:07:04 Tuesday询问阿哲是否没吃午饭。",
    "阿哲在2025-09-30 15:08:45 Tuesday表示因会议持续到14:00而未吃午饭，只喝了咖啡。",
    "我在2025-09-30 15:09:01 Tuesday提醒阿哲注意胃病。",
    "阿哲在2025-09-30 15:09:42 Tuesday向我解释因老板临时要求修改PPT至第8版而没吃饭。",
    "我在2025-09-30 15:10:04 Tuesday提出晚上给阿哲带皮蛋瘦肉粥。",
    "阿哲在2025-09-30 15:10:22 Tuesday承诺请我喝一个月奶茶。",
    "我在2025-09-30 15:11:46 Tuesday告知阿哲已将粥放在其工位上。",
    "阿哲在2025-09-30 15:12:01 Tuesday确认看到我放在他工位上的粥。",
    "阿哲在2025-09-30 15:12:09 Tuesday向我表达感谢。"
  ]
}}
</output>
</example>'''
        }
    },
    'required': ['episodic_memories']
}

extract_reflective_memories_schema = {
    'title': 'extract_reflective_memories',
    'type': 'object',
    'description': f"""请使用这个工具来返回你从记录中提取到的记忆信息，请以第一人称视角提取记忆信息，提取到的所有记忆信息需满足以下规则：
1. 代词转换为全称：
- 在你最终输出的信息里出现的"我"只能指代当前用户（即需要提取记忆信息的主体），这是为了保持第一人称视角
- 除"我"外，所有代词（你/他/她/他们）必须替换为完整姓名
- 例：原句"他昨天去了超市。" → "李四昨天去了超市。
2. 时间规则：
- 不要出现精确时间，但可在句子中体现“曾”、“经常”等时态
- 如果确有必要，请遵守以下关于精确时间的规范：
    - 使用%Y-%m-%d %H:%M:%S %A格式，如2025-11-30 09:30:00 Sunday
    - 尽可能精确到秒和周几，但原始记录中没有具体信息时也可以从后往前省略一些时间信息，或直接使用文字代替
    - 例：原句"上周三早上" → "2024-03-20 早上 Wednesday"

<example>
<input>
<persona>
{example_character}
</persona>
<history>
{example_history}
</history>
</input>
<output>
{{
  "reflection_process": "阿哲今天又没吃午饭，只靠咖啡撑着，这让我立刻想到他上次胃疼的事——显然他的胃疼和不规律饮食有关。他总说“本来想吃”“下次一定”，但每次都被工作打断，说明这不是偶然，而是他的工作模式导致长期饮食不规律。而且他这次又承诺请我喝一个月奶茶，可上次说请吃饭到现在都没兑现，明显是习惯性‘画饼’。尽管如此，他收到粥时特别激动，还说‘爱你’，说明他其实很在意我的关心，只是表达方式比较贫。我的话虽然嘴上骂他，但看到他这么累还是忍不住帮他——可能是因为我知道他本性不坏，只是太忙了。综合来看，他对工作投入过度，健康堪忧。",
  "reflective_memories": [
    "阿哲时常因不规律饮食而胃疼。",
    "阿哲经常因高强度工作而无法按时用餐，形成固定行为模式。",
    "阿哲习惯做出无法兑现的承诺（如请吃饭、请喝奶茶、不加班），属于‘画饼型’沟通风格。",
    "我对阿哲的健康状况持续担忧，并会主动提供实际帮助（如送粥）。",
    "阿哲在感受到关心时会真诚表达感激，情感反馈积极。",
    "我和阿哲的关系表现为表面互怼、实则互相关心的亲密互动模式。",
    "阿哲的工作压力极大，常需反复修改方案（如PPT改到第8版以上）。",
    "我理解阿哲的处境，并对他保持包容与情感支持。"
  ]
}}
</output>
</example>""",
    'properties': {
        'reflection_process': {
            #'title': 'Reflection Process',
            'type': 'string',
            'description': '代入进角色视角展开的第一人称的思考及推理过程。根据角色设定，可能这是一个很认真的人，会认真地思考每个细节，甚至反思自己的情感（很多人会更看重他人的看法，这样往往会忽略自己的情感）；也可能是一个很懒的人，根本懒得去想那么多。千人千面，总之要根据角色设定输出“看起来像是这个角色会想的东西”的内容。'
        },
        'reflective_memories': {
            'type': 'array',
            'items': {'type': 'string'},
            'description': '''经过思考后得出的原子级反思结论或猜测以及依据。提取出的原子级信息除了需要满足刚才提到的代词转换规则，还需满足：
1. 使用尽可能简洁的SPO句式：
- 主语(S)：专有名词或"我"
    专有名词定义：
    - 包含人名/地名/机构名/特定事件名等
    - 例："李四"、"上海交通大学"、"2024春季运动会"
- 谓语(P)：动词/形容词/系动词
- 宾语(O)：所有名词/数值/时间等
2. 原子性要求：
- 单句仅表达一个结论或猜测，以及得出该结论或猜测的依据'''
        }
    },
    'required': ['reflection_process', 'reflective_memories']
}



async def recycle_original_memories(agent_id: str, input_messages: list[AnyMessage]):
    store_settings = await store_manager.get_settings(agent_id)
    time_settings = store_settings.main.time_settings
    messages = filtering_messages(input_messages, exclude_extracted=False)
    content_and_kwargs: list[dict[str, Any]] = []
    formated_messages = format_messages_for_ai_as_list(messages, time_settings)
    index = 0
    for m in messages:
        if isinstance(m, (HumanMessage, AIMessage)):
            content_and_kwargs.append({'content': formated_messages[index], 'kwargs': m.additional_kwargs, 'id': m.id})
            index += 1

    extracted_memories: list[InitialMemory] = []
    current_agent_timeseconds = now_agent_seconds(time_settings)
    base_stable_time = store_settings.recycling.base_stable_time
    message_ids = [message['id'] if message['id'] else str(uuid4()) for message in content_and_kwargs]
    messages_len = len(content_and_kwargs)

    for i, message in enumerate(content_and_kwargs):
        stable_mult = random.expovariate(0.8) #TODO:这个值应该由文本的情感强烈程度来决定
        creation_agent_time_seconds = message['kwargs'].get("bh_creation_agent_timeseconds", current_agent_timeseconds)
        creation_agent_datetime = agent_seconds_to_datetime(creation_agent_time_seconds, time_settings)
        extracted_memories.append(InitialMemory(
            content=message['content'],
            stable_time=stable_mult * base_stable_time,
            type="original",
            creation_agent_datetime=creation_agent_datetime,
            id=message_ids[i],
            previous_memory_id=None if i == 0 else message_ids[i-1],
            next_memory_id=None if i == messages_len - 1 else message_ids[i+1]
        ))

    await memory_manager.add_memories(extracted_memories, agent_id)


async def recycle_episodic_memories(agent_id: str, input_messages: list[AnyMessage], model: BaseChatModel):
    messages = filtering_messages(input_messages)

    extracted_memories = []

    store_settings = await store_manager.get_settings(agent_id)
    time_settings = store_settings.main.time_settings
    current_agent_time_seconds = now_agent_seconds(time_settings)
    base_stable_time = store_settings.recycling.base_stable_time

    creation_timesecondses = [int(m.additional_kwargs.get("bh_creation_agent_timeseconds", current_agent_time_seconds)) for m in messages]

    if messages:
        # 过渡方案，对于总结和语义记忆的creation_time_seconds直接使用原始记录的平均时间戳
        creation_timeseconds_average = sum(creation_timesecondses) / len(creation_timesecondses)
        creation_datetime_average = agent_seconds_to_datetime(creation_timeseconds_average, time_settings)

        llm_with_structure = create_agent(model, response_format=extract_episodic_memories_schema)
        extracted_episodic_memories = (await llm_with_structure.ainvoke({'messages': [HumanMessage(content=f"""以下是用户的最近记录：
<history>
{format_messages_for_ai(messages, time_settings)}
</history>
请你将这些XML格式的记录根据要求分解为 episodic memories。"""
        )]}))['structured_response']

        episodic_memories = extracted_episodic_memories["episodic_memories"]
        episodic_memory_ids = [str(uuid4()) for _ in episodic_memories]
        episodic_memories_len = len(episodic_memories)

        for i, episodic_memory in enumerate(episodic_memories):
            extracted_memories.append(InitialMemory(
                content=episodic_memory,
                stable_time=random.expovariate(1.0) * base_stable_time,
                type="episodic",
                creation_agent_datetime=creation_datetime_average,
                id=episodic_memory_ids[i],
                previous_memory_id=None if i == 0 else episodic_memory_ids[i-1],
                next_memory_id=None if i == episodic_memories_len - 1 else episodic_memory_ids[i+1]
            ))

    await memory_manager.add_memories(extracted_memories, agent_id)


async def recycle_reflective_memories(agent_id: str, input_messages: list[AnyMessage], model: BaseChatModel) -> list[BaseMessage]:
    messages = filtering_messages(input_messages)
    store_settings = await store_manager.get_settings(agent_id)
    times_before = Times(store_settings.main.time_settings)
    parsed_character_settings = store_settings.main.format_character_settings()
    role_prompt = f'基本信息：\n{parsed_character_settings if parsed_character_settings.strip() else '无'}\n\n详细设定：\n{store_settings.main.role_prompt}'
    #llm_with_structure = self.llm.with_structured_output(ExtractReflectiveMemories, method="function_calling")
    llm_with_structure = create_agent(
        model,
        response_format=ToolStrategy(extract_reflective_memories_schema),
    )
    extracted_reflective_memories = (await llm_with_structure.ainvoke({"messages": [HumanMessage(content=f"""以下是用户的角色设定：
<persona>
{role_prompt}
</persona>

以下是用户的最近记录（XML格式）：
<history>
{format_messages_for_ai(messages, store_settings.main.time_settings)}
</history>
请你以此角色的视角，思考记录中发生的事意味着什么，能得出什么样的结论或猜测，然后输出要求的 reflection process 和 reflective memories。""")]}))["structured_response"]
    #dumped_reflective_memories = extracted_reflective_memories.model_dump()
    reflective_memories = extracted_reflective_memories["reflective_memories"]
    ids = [str(uuid4()) for _ in reflective_memories]
    reflective_memories_len = len(reflective_memories)
    times_after = Times(store_settings.main.time_settings)
    process: list[BaseMessage] = []
    process.append(construct_system_message(
        content=f'''当前时间是 {format_time(times_before.agent_datetime)}，距上次与用户聊天过去了一段时间，现在是一个反思刚才所发生的事情的好时机。请你以你所扮演的角色的视角，思考刚才所发生的事意味着什么，能得出什么样的结论或猜测，得出结果并留下思考过程。''',
        times=times_before
    ))
    process.extend(InitalAIMessage(
        tool_calls=[InitalToolCall(
            name=RECORD_THOUGHTS,
            args={RECORD_THOUGHTS_CONTENT: extracted_reflective_memories["reflection_process"]},
            result_content=RECORD_THOUGHTS_TOOL_CONTENT,
            result_artifact={"bh_do_not_store": True, "bh_streaming": True}
        )]
    ).construct_messages(times_after))
    base_stable_time = store_settings.recycling.base_stable_time
    memories = [InitialMemory(
        content=memory,
        stable_time=random.expovariate(0.4) * base_stable_time,
        type="reflective",
        creation_agent_datetime=times_after.agent_datetime,
        id=ids[i],
        previous_memory_id=None if i == 0 else ids[i-1],
        next_memory_id=None if i == reflective_memories_len - 1 else ids[i+1]
    ) for i, memory in enumerate(reflective_memories)]
    if 'original' in get_activated_memory_types():
        memories.append(InitialMemory(
            content=process[1].content,
            stable_time=random.expovariate(0.4) * base_stable_time,
            type="original",
            creation_agent_datetime=times_after.agent_datetime,
            id=process[1].id,
            previous_memory_id=None,
            next_memory_id=None
        ))
        for m in process:
            m.additional_kwargs['bh_recycled'] = True
    await memory_manager.add_memories(memories, agent_id)
    return process


async def recycle_memories(memory_type: AnyMemoryType, agent_id: str, input_messages: list[AnyMessage], model: Optional[BaseChatModel] = None) -> Optional[list[BaseMessage]]:
    if model is None and (memory_type == "reflective" or memory_type == "episodic"):
        raise ValueError("model is required for episodic or reflective memories")
    elif memory_type == "original":
        return await recycle_original_memories(agent_id, input_messages)
    elif memory_type == "episodic":
        return await recycle_episodic_memories(agent_id, input_messages, model)
    elif memory_type == "reflective":
        return await recycle_reflective_memories(agent_id, input_messages, model)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
