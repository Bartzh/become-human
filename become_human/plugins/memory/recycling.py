from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import uuid4
from loguru import logger
import random
import asyncio

from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, BaseMessage
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import BaseChatModel
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from become_human.utils import gather_safe
from become_human.times import format_time, Times, TimestampUs, get_week
from become_human.store.manager import store_manager
from become_human.event import event_bus
from become_human.message import (
    construct_system_message,
    SpritedMsgMeta,
    InitalAIMessage,
    InitalToolCall,
    SpritedMsgMetaOptionalTimes,
)
from become_human.tools.send_message import SEND_MESSAGE, SEND_MESSAGE_CONTENT, SEND_MESSAGE_TOOL_CONTENT
from become_human.tools.record_thoughts import RECORD_THOUGHTS, RECORD_THOUGHTS_CONTENT, RECORD_THOUGHTS_TOOL_CONTENT
from become_human.manager import sprite_manager
from become_human.plugins.memory.base import (
    AnyMemoryType,
    memory_manager,
    InitialMemory,
)
from become_human.plugins.memory.store import MemoryConfig, MemoryData
from become_human.plugins.memory.message import filtering_messages, format_messages_for_ai, MemoryMsgMeta, format_messages_for_ai_as_list


rmprompt = '0. 这份记录实际上来自一个AI，但请把它当作人类来撰写你提取到的信息，不要暴露这个人是一个AI的事实'
example_character = '你是一个关心他人、略带毒舌但内心柔软的职场年轻人，性别女性，25岁。'
# 暂时没有使用
example_summary_output = {
    'year': '''{
  "summary_content": "2024 年，我经历了工作上的项目起起落落，从春季的新产品发布到秋季的技术重构，见证了团队的成长。与阿哲的互动频繁，他多次做出无法兑现的承诺，但彼此间的默契加深。健康方面，开始注重规律作息，偶尔仍有熬夜赶工的情况。这一年中，逐渐学会平衡工作与生活，意识到在关心他人的同时也要照顾好自己。"
}''',
    'month': '''{
  "summary_content": "本月工作节奏加快，临近年底各项目赶进度。与阿哲的互动频率增加，多次提醒其按时用餐。月中参加团建活动，团队成员得以放松。月底出现感冒症状，注意到换季时需关注身体状况。整体而言，本月工作忙碌但充实。"
}''',
    'week': '''{
  "summary_content": "本周工作强度适中，周一至周三处理日常事务，周四至周五讨论下季度规划。阿哲三天未按时吃午饭，靠咖啡维持，我多次提醒并提供饮食支持。周末参加朋友生日聚会，获得短暂放松。本周整体平稳，与阿哲的互动成为重点。"
}''',
    'day': '''{
  "summary_content": "今日工作忙碌，上午处理紧急需求，下午参加部门会议。阿哲未吃午饭，靠咖啡维持，我点粥送至其工位。阿哲收到粥后表现出惊喜和感激。晚上加班至八点，回家路上购买奶茶。今日情绪因阿哲的状况出现波动，后因帮助他人而感到满足。"
}'''
}
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
- 使用%Y-%m-%d %H:%M:%S Week%W %A格式，如2025-11-30 09:30:00 Week47 Sunday
- 尽可能精确到秒、第几周和周几，但原始记录中没有具体信息时也可以从后往前省略一些时间信息，或直接使用文字代替
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
    - 使用%Y-%m-%d %H:%M:%S Week%W %A格式，如2025-11-30 09:30:00 Week47 Sunday
    - 尽可能精确到秒、周数和周几，但原始记录中没有具体信息时也可以省略一些时间信息，或直接使用文字代替
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

summary_memories_schema = {
    'title': 'summary_memories',
    'type': 'object',
    # 暂时不用示例
    'description': f"""请使用这个工具来返回你对原始记忆的总结内容。总结需要以第一人称视角撰写，并满足以下规则：
1. 叙述风格：
- 客观陈述，不要艺术加工
- 清晰简洁，突出重要事件和情绪波动
- 保持第一人称视角，但避免主观评价和情感渲染
- 客观记录你自己做过或想过的事实以及情感状态
2. 内容优先级：
- 优先记录对自己重要的事件
- 优先记录给自己留下深刻印象的事件
- 优先记录引起情绪波动的事件
3. 代词规则：
- 人名应在尽可能的情况下使用具体姓名，除非不知道
- 唯一例外是，请保持使用"我"来指代你自己（为了保持第一人称视角）""",
    'properties': {
        'summary_content': {
            'type': 'string',
            'description': '对给定时间范围内所有原始记忆的综合总结内容。应该是一段客观、清晰的陈述文字，优先记录重要事件和情绪波动，涵盖工作、人际关系、个人成长、健康状况等方面。'
        }
    },
    'required': ['summary_content']
}



async def connect_last_memory(sprite_id: str, memory_type: AnyMemoryType, new_memory_ids: list[str]) -> Optional[str]:
    data_store = store_manager.get_model(sprite_id, MemoryData)
    last_added_memory_ids = data_store.last_added_memory_ids
    if last_added_memory_ids.get(memory_type):
        last_id = last_added_memory_ids[memory_type]
        await memory_manager.update_metadatas(
            ids=[last_id],
            metadatas=[{'next_memory_id': new_memory_ids[0]}],
            memory_type=memory_type,
            sprite_id=sprite_id
        )
    else:
        last_id = None
    last_added_memory_ids[memory_type] = new_memory_ids[-1]
    data_store.last_added_memory_ids = last_added_memory_ids
    return last_id

async def recycle_original_memories(sprite_id: str, input_messages: list[AnyMessage]):
    config_store = store_manager.get_model(sprite_id, MemoryConfig)
    messages = filtering_messages(input_messages, exclude_extracted=False)
    content_and_kwargs: list[dict[str, Any]] = []
    formated_messages = format_messages_for_ai_as_list(messages)
    content_and_kwargs = [{'content': s, 'kwargs': messages[i].additional_kwargs, 'id': messages[i].id} for s, i in formated_messages]

    extracted_memories: list[InitialMemory] = []
    base_ttl = config_store.memory_base_ttl
    message_ids = [message['id'] if message['id'] else str(uuid4()) for message in content_and_kwargs]
    messages_len = len(content_and_kwargs)

    if messages_len == 0:
        return

    last_id = await connect_last_memory(sprite_id, 'original', message_ids)

    for i, message in enumerate(content_and_kwargs):
        stable_mult = random.expovariate(0.8) #TODO:这个值应该由文本的情感强烈程度来决定
        extracted_memories.append(InitialMemory(
            content=message['content'],
            ttl=int(stable_mult * base_ttl),
            type="original",
            creation_times=SpritedMsgMeta.parse(message['kwargs']).creation_times,
            id=message_ids[i],
            previous_memory_id=last_id if i == 0 else message_ids[i-1],
            next_memory_id=None if i == messages_len - 1 else message_ids[i+1]
        ))

    await memory_manager.add_memories(extracted_memories, sprite_id)


async def recycle_episodic_memories(sprite_id: str, input_messages: list[AnyMessage], model: BaseChatModel):
    messages = filtering_messages(input_messages)

    if messages:
        extracted_memories = []

        config_store = store_manager.get_model(sprite_id, MemoryConfig)
        store_settings = store_manager.get_settings(sprite_id)
        time_settings = store_settings.time_settings
        base_ttl = config_store.memory_base_ttl

        # 目前就用当前时间了
        current_times = Times.from_time_settings(time_settings)

        llm_with_structure = create_agent(model, response_format=extract_episodic_memories_schema)
        extracted_episodic_memories = (await llm_with_structure.ainvoke({'messages': [HumanMessage(content=f"""以下是用户的最近记录：
<history>
{format_messages_for_ai(messages)}
</history>
请你将这些XML格式的记录根据要求分解为 episodic memories。"""
        )]}))['structured_response']

        episodic_memories = extracted_episodic_memories["episodic_memories"]
        episodic_memory_ids = [str(uuid4()) for _ in episodic_memories]
        episodic_memories_len = len(episodic_memories)

        if episodic_memories_len == 0:
            logger.warning(f"{sprite_id} 没有提取到任何 episodic memories")
            return

        last_id = await connect_last_memory(sprite_id, 'episodic', episodic_memory_ids)

        for i, episodic_memory in enumerate(episodic_memories):
            extracted_memories.append(InitialMemory(
                content=episodic_memory,
                ttl=int(random.expovariate(1.0) * base_ttl),
                type="episodic",
                creation_times=current_times,
                id=episodic_memory_ids[i],
                previous_memory_id=last_id if i == 0 else episodic_memory_ids[i-1],
                next_memory_id=None if i == episodic_memories_len - 1 else episodic_memory_ids[i+1]
            ))

        await memory_manager.add_memories(extracted_memories, sprite_id)


async def recycle_reflective_memories(sprite_id: str, input_messages: list[AnyMessage], model: BaseChatModel) -> list[BaseMessage]:
    messages = filtering_messages(input_messages)
    if not messages:
        return
    store_settings = store_manager.get_settings(sprite_id)
    times_before = Times.from_time_settings(store_settings.time_settings)
    parsed_character_settings = store_settings.format_character_settings()
    role_prompt = f'基本信息：\n{parsed_character_settings if parsed_character_settings.strip() else '无'}\n\n详细设定：\n{store_settings.role_prompt}'
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
{format_messages_for_ai(messages)}
</history>
请你以此角色的视角，思考记录中发生的事意味着什么，能得出什么样的结论或猜测，然后输出要求的 reflection process 和 reflective memories。""")]}))["structured_response"]
    #dumped_reflective_memories = extracted_reflective_memories.model_dump()
    reflective_memories = extracted_reflective_memories.get("reflective_memories", [])
    ids = [str(uuid4()) for _ in reflective_memories]
    reflective_memories_len = len(reflective_memories)
    times_after = Times.from_time_settings(store_settings.time_settings)
    base_ttl = store_manager.get_model(sprite_id, MemoryConfig).memory_base_ttl
    memories = []

    if reflective_memories_len > 0:
        last_id = await connect_last_memory(sprite_id, 'reflective', ids)
        memories.extend([InitialMemory(
            content=memory,
            ttl=int(random.expovariate(0.4) * base_ttl),
            type="reflective",
            creation_times=times_after,
            id=ids[i],
            previous_memory_id=last_id if i == 0 else ids[i-1],
            next_memory_id=None if i == reflective_memories_len - 1 else ids[i+1]
        ) for i, memory in enumerate(reflective_memories)])
    else:
        logger.warning(f"{sprite_id} 没有提取到任何 reflective memories")

    process: list[BaseMessage] = []
    if extracted_reflective_memories.get("reflection_process"):
        process.append(construct_system_message(
            content=f'''当前时间是 {format_time(times_before.sprite_world_datetime)}，距上次与用户聊天过去了一段时间，现在是一个反思刚才所发生的事情的好时机。请你以你所扮演的角色的视角，思考刚才所发生的事意味着什么，能得出什么样的结论或猜测，得出结果并留下思考过程。''',
            times=times_before
        ))
        process.extend(InitalAIMessage(
            tool_calls=[InitalToolCall(
                name=RECORD_THOUGHTS,
                args={RECORD_THOUGHTS_CONTENT: extracted_reflective_memories["reflection_process"]},
                result_content=RECORD_THOUGHTS_TOOL_CONTENT,
                result_msg_metas=[
                    SpritedMsgMetaOptionalTimes(
                        is_action_only_tool=True
                    )
                ]
            )]
        ).construct_messages(times_after))
        if 'original' in store_manager.get_model(sprite_id, MemoryConfig).memory_types:
            last_id = await connect_last_memory(sprite_id, 'original', [process[1].id])
            memories.append(InitialMemory(
                content=process[1].content,
                ttl=int(random.expovariate(0.4) * base_ttl),
                type="original",
                creation_times=times_after,
                id=process[1].id,
                previous_memory_id=last_id,
                next_memory_id=None
            ))
            recycled_meta = MemoryMsgMeta(recycled=True)
            for m in process:
                recycled_meta.update_to(m)
    else:
        logger.warning(f"{sprite_id} 没有提取到任何 reflection process")

    if memories:
        await memory_manager.add_memories(memories, sprite_id)
    return process

async def recycle_summary_memories(sprite_id: str, model: BaseChatModel) -> None:
    """应在每次sleeping时调用"""
    # TODO
    # 首先，应检查需要总结哪些时间范围的记忆（年、月、周、日）
    store_settings = store_manager.get_settings(sprite_id)
    time_settings = store_settings.time_settings
    current_times = Times.from_time_settings(time_settings)
    current_datetime = current_times.sprite_world_datetime
    data_store = store_manager.get_model(sprite_id, MemoryData)
    last_summarized_times = data_store.last_summarized_times
    if not last_summarized_times:
        data_store.last_summarized_times = current_times
        logger.info(f"{sprite_id} 没有上次总结时间，将设置当前时间为新的最后总结时间")
        return
    last_summarized_times = last_summarized_times.model_copy()
    last_datetime = last_summarized_times.sprite_world_datetime

    data_store.last_summarized_times = current_times

    config_store = store_manager.get_model(sprite_id, MemoryConfig)
    # datetime在进行比较时会将时间转换至UTC
    if last_datetime > current_datetime:
        logger.warning(f"{sprite_id} 上次总结时间晚于当前时间，可能进行了时空穿越回到了过去，将跳过总结并设置当前时间为新的最后总结时间")
        return
    elif last_datetime == current_datetime:
        logger.warning(f"{sprite_id} 上次总结时间与当前时间相同，可能只是因为时区发生了改变，将跳过总结并设置当前时间为新的最后总结时间")
        return

    def get_season(month: int) -> int:
        """根据月份返回季节 (1=春，2=夏，3=秋，4=冬)"""
        if month in [3, 4, 5]:
            return 1  # 春季
        elif month in [6, 7, 8]:
            return 2  # 夏季
        elif month in [9, 10, 11]:
            return 3  # 秋季
        else:  # 12, 1, 2
            return 4  # 冬季

    def get_season_start_month(season: int) -> int:
        """根据季节返回起始月份"""
        season_start_map = {1: 3, 2: 6, 3: 9, 4: 12}
        return season_start_map[season]

    def get_season_start_datetime(dt: datetime) -> datetime:
        """获取当前季节的起始时间"""
        season = get_season(dt.month)
        start_month = get_season_start_month(season)
        # 如果当前月份小于起始月份，说明是上一年的季节（如 1-2 月属于冬季，起始是 12 月）
        if dt.month < start_month:
            return datetime(dt.year - 1, start_month, 1)
        else:
            return datetime(dt.year, start_month, 1)


    exclude_granularities = []
    if abs(current_datetime.year - last_datetime.year) >= 2:
        # 年的差异大于等于2年，则任何粒度都要进行总结
        pass
    else:
        if (
            get_week(current_datetime) == get_week(last_datetime) or
            (
                get_week(current_datetime) == 0 and
                get_week(last_datetime) == get_week(last_datetime.replace(month=12, day=31))
            )
        ):
            exclude_granularities.append('week')
        # 判断季节是否变化
        if get_season(current_datetime.month) == get_season(last_datetime.month):
            exclude_granularities.append('season')
        if current_datetime.year != last_datetime.year:
            # 年发生变动，则年月日都要进行总结
            pass
        else:
            exclude_granularities.append('year')
            if current_datetime.month == last_datetime.month:
                exclude_granularities.append('month')
            if (
                current_datetime.day == last_datetime.day and
                'month' in exclude_granularities
            ):
                exclude_granularities.append('day')

    time_granularities = [g for g in config_store.summary_time_granularities if g not in exclude_granularities]
    if not time_granularities:
        logger.info(f"{sprite_id} 不需要进行任何总结")
        return


    def _clac_all_timestampus(dt: datetime) -> dict[str, TimestampUs]:
        """计算所有时间粒度的时间戳"""
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return {
            'year': TimestampUs(datetime(dt.year, 1, 1)),
            'season': TimestampUs(get_season_start_datetime(dt)),
            'month': TimestampUs(datetime(dt.year, dt.month, 1)),
            'week': TimestampUs(dt - timedelta(days=dt.weekday())),
            'day': TimestampUs(dt),
        }

    granularities_last = _clac_all_timestampus(last_datetime)
    granularities_current = _clac_all_timestampus(current_datetime)


    base_ttl = config_store.memory_base_ttl
    # 获取角色设定
    parsed_character_settings = store_settings.format_character_settings()
    role_prompt = f'基本信息：\n{parsed_character_settings if parsed_character_settings.strip() else '无'}\n\n详细设定：\n{store_settings.role_prompt}'

    async def _extract(granularity: str) -> Optional[InitialMemory]:
        # 然后get所有需要总结的（original）记忆，并给它们进行recall
        last = granularities_last[granularity]
        current = granularities_current[granularity]
        recall_strength = 0.7 if granularity != 'day' else 3.0 #TODO 配置

        if granularity == 'day':
            creation_time_where = {
                '$and': [
                    {'creation_sprite_world_timestampus': {
                        '$gte': last
                    }},
                    {'creation_sprite_world_timestampus': {
                        '$lt': current
                    }},
                    {'retrievability': {
                        '$gt': 0.15
                    }}
                ]
            }
            docs = await memory_manager.get_memories(
                sprite_id=sprite_id,
                memory_type="original",
                where=creation_time_where,
                strength=recall_strength
            )
            docs += await memory_manager.get_memories(
                sprite_id=sprite_id,
                memory_type="reflective",
                where=creation_time_where,
                strength=recall_strength
            )
        else:
            docs = await memory_manager.get_memories(
                sprite_id=sprite_id,
                memory_type="summary",
                where={
                    '$and': [
                        {'summary_time_granularity': granularity},
                        {'summary_timestampus': {
                            '$gte': last
                        }},
                        {'summary_timestampus': {
                            '$lt': current
                        }},
                        {'retrievability': {
                            '$gt': 0.15
                        }}
                    ]
                },
                strength=recall_strength
            )
        if not docs:
            return None

        docs.sort(key=lambda x: x.metadata['retrievability'], reverse=True)
        total_chars = 0
        # 保守估计token数
        # TODO 配置选项，目前最大十万token
        max_chars = int(100000 * 1.5)
        include_idx = []
        for i, doc in enumerate(docs):
            total_chars += len(doc.page_content)
            if total_chars > max_chars:
                total_chars -= len(doc.page_content)
                break
            include_idx.append(i)
        docs: list[Document] = [docs[i] for i in include_idx]
        docs.sort(key=lambda x: x.metadata['creation_sprite_world_timestampus'])

        # 计算输入记忆的总字数
        # 根据输入字数确定总结长度
        if total_chars < 2000:
            summary_length = 500
        else:
            summary_length = 1000

        # 不同时间粒度的内容提示方向
        granularity_guidelines = {
            'year': '这是年度总结，应关注这一年中经历的重要事件、生活变化、人际关系和成长历程，记录随着时间推移发生的各种故事和感受。',
            'season': '这是季度总结，应关注这三个月内经历的重要事件、生活变化、人际关系和个人体验，记录随着时间推移发生的各种故事。',
            'month': '这是月度总结，应关注这个月里发生的事情、人际关系和情绪波动。',
            'week': '这是周度总结，应记录这周发生的具体事件、与人互动和情绪波动，记录一周的生活轨迹。',
            'day': '这是日度总结，应记录今天的具体经历、与人对话、饮食起居、身体健康状况和情绪感受，记录一天的生活轨迹。'
        }

        # 构建prompt
        prompt = f"""请你根据你自己的角色设定，提要总结你自己在时间范围`{granularity}`内的经历，以下是该任务的详细说明：

# 角色设定
{role_prompt}

# 要求

## 时间范围说明
{granularity_guidelines[granularity]}

## 内容长度
{summary_length}字左右的总结内容，保持客观陈述风格。
{'''
# 记忆格式说明
可能出现为XML格式的记忆，包含me标签、action标签和others标签，记录了具体的行为动作。
- <me>标签：记录自己的行为，包含时间和内容
    - <action>标签：记录自己的行为的具体动作名称和内容
- <others>标签：记录他人的行为，包含时间和内容
''' if granularity == 'day' else ''}
# 将要进行提要总结记忆内容
以下是你自己在时间范围 “{granularity}” 内的记忆，按时间顺序排列：

<memories>
{'\n\n\n'.join([doc.page_content for doc in docs])}
</memories>

请你根据这些记忆，结合角色设定（persona），按照`summary_memories`要求的规则撰写{granularity}总结。优先记录对自己重要的事件、留下深刻印象的事件和引起情绪波动的事件。"""

        # 最后，调用模型总结记忆
        llm_with_structured = create_agent(
            model,
            response_format=ToolStrategy(summary_memories_schema)
        )
        response = await llm_with_structured.ainvoke({'messages': [HumanMessage(content=prompt)]})
        result = response['structured_response'].get('summary_content')
        if not result:
            logger.error(f"{sprite_id} 总结记忆 {granularity} 时，模型返回空结果")
            return None
        ttl_mult_map = {
            'year': 5,
            'season': 3,
            'month': 2,
            'week': 1.5,
            'day': 1,
        }
        return InitialMemory(
            content=f'{format_time(
                last,
                last_summarized_times.sprite_time_settings.time_zone
            )} ~ {format_time(
                current,
                last_summarized_times.sprite_time_settings.time_zone
            )}\n' + result,
            ttl=int(random.expovariate(0.3) * base_ttl * ttl_mult_map[granularity]),
            type=f"summary",
            creation_times=Times.from_time_settings(store_manager.get_settings(sprite_id).time_settings),
            extra={
                'summary_time_granularity': granularity,
                'summary_timestampus': last,
                'summary_date_iso': last.to_datetime().date().isoformat(),
            },
            # 暂时不考虑连接
            # id=None,
            # previous_memory_id=None,
            # next_memory_id=None
        )

    tasks = [_extract(g) for g in time_granularities]
    memories = await gather_safe(*tasks)
    memories = [m for m in memories if m is not None]
    await memory_manager.add_memories(memories, sprite_id)


async def recycle_memories(memory_type: AnyMemoryType, sprite_id: str, input_messages: list[AnyMessage], model: Optional[BaseChatModel] = None) -> Optional[list[BaseMessage]]:
    if model is None and (memory_type == "reflective" or memory_type == "episodic"):
        raise ValueError("model is required for episodic or reflective memories")
    elif memory_type == "original":
        return await recycle_original_memories(sprite_id, input_messages)
    elif memory_type == "episodic":
        return await recycle_episodic_memories(sprite_id, input_messages, model)
    elif memory_type == "reflective":
        return await recycle_reflective_memories(sprite_id, input_messages, model)
    elif memory_type == "summary":
        return await recycle_summary_memories(sprite_id, model)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")



@event_bus.on('bh_presence:on_presence_changed')
async def on_sprite_away_or_sleeping(sprite_id: str, new: Any) -> None:
    if new.is_sleeping():
        data_store = store_manager.get_model(sprite_id, MemoryData)
        # 目前就这样，在睡觉时清空字典，切断记忆连接
        data_store.last_added_memory_ids = {}
        await recycle_summary_memories(sprite_id, sprite_manager.structured_model)
    elif not new.is_away():
        return
    messages = await sprite_manager.get_messages(sprite_id)
    # 最后一条消息如果为HumanMessage说明sprite还没有响应
    if not isinstance(messages[-1], HumanMessage):
        default = MemoryMsgMeta()
        not_extracted_messages = [m for m in messages if not default.parse_with_default(m).extracted]
        # 再次判断是否有来自用户的新消息
        if sprite_manager.get_plugin('bh_presence').is_user_input(not_extracted_messages):
            return

        config_store = store_manager.get_model(sprite_id, MemoryConfig)
        remove_messages = []

        # recycling
        memory_types = config_store.memory_types
        recycles = {t: recycle_memories(t, sprite_id, not_extracted_messages, sprite_manager.structured_model) for t in memory_types}
        recycle_results = {}
        if len(recycles) > 0:
            graph_results = await gather_safe(*recycles.values())
            recycle_results = {k: graph_results[i] for i, k in enumerate(recycles.keys())}

        # cleanup
        is_cleanup = config_store.cleanup_on_unavailable
        if is_cleanup:
            #await main_graph.update_messages(sprite_id, [RemoveMessage(id=m.id) for m in messages])
            max_tokens = config_store.cleanup_target_size
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
        extracted_meta = MemoryMsgMeta(extracted=True)
        for m in not_extracted_messages:
            extracted_meta.update_to(m)
        await sprite_manager.update_messages(sprite_id, not_extracted_messages + remove_messages)

        # 若有，将reflective的思考过程加入messages
        if recycle_results.get('reflective'):
            await sprite_manager.update_messages(sprite_id, recycle_results.get('reflective', []))
