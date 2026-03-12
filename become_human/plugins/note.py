from typing import Annotated, Optional, override
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from become_human.store.base import StoreModel, StoreField
from become_human.store.manager import store_manager
from become_human.manager import CallSpriteRequest
from become_human.plugin import *
from become_human.manager import sprite_manager
from become_human.times import Times, format_time
from become_human.message import DictMsgMeta

NAME = 'note'

@dataclass
class Note:
    title: str
    content: str

class NoteData(StoreModel):
    _namespace = NAME + '_data'
    notes: dict[int, Note] = StoreField(default_factory=dict)
    next_id: int = StoreField(default=0)


@tool(response_format='content_and_artifact')
async def list_notes(runtime: ToolRuntime[CallSpriteRequest]) -> str:
    """列出所有笔记（的标题）"""
    notes = store_manager.get_model(runtime.context.sprite_id, NoteData).notes
    if not notes:
        return "暂无任何笔记"
    artifact = DictMsgMeta(
        KEY='bh_memory',
        value={
            'do_not_store': True,
        }
    )
    return "\n".join([f"{note_id}. {note.title}" for note_id, note in notes.items()]), artifact

@tool(response_format='content_and_artifact')
async def read_note(runtime: ToolRuntime[CallSpriteRequest], id: Annotated[int, "笔记ID"]) -> str:
    """读取指定笔记"""
    notes = store_manager.get_model(runtime.context.sprite_id, NoteData).notes
    if not notes:
        return "暂无任何笔记"
    note = notes.get(id)
    if note is None:
        return f"不存在ID为{id}的笔记"
    artifact = DictMsgMeta(
        KEY='bh_memory',
        value={
            'do_not_store': True,
        }
    )
    return f'笔记内容：{note.content}', artifact

@tool
async def write_note(
    runtime: ToolRuntime[CallSpriteRequest],
    title: Annotated[str, "笔记标题"],
    content: Annotated[str, "笔记内容"],
    id: Annotated[Optional[int], "指定笔记ID。这只适用于想要覆盖已存在的笔记的情况，如果不存在该ID的笔记，将不做修改直接返回"] = None
) -> str:
    """写入笔记"""
    if not title.strip() or not content.strip():
        return "笔记标题或内容不能为空"
    sprite_id = runtime.context.sprite_id
    data = store_manager.get_model(sprite_id, NoteData)
    if id or id == 0:
        try:
            id = int(id)
        except Exception:
            raise ValueError("输入的笔记ID不是一个整数")
        if id < 0:
            raise ValueError("笔记ID不能为负数")
        if id not in data.notes:
            return f"不存在ID为{id}的笔记" 
        else:
            output = f"覆盖笔记成功"
    else:
        id = data.next_id
        data.next_id += 1
        output = "新增笔记成功"
    notes = data.notes.copy()
    notes[id] = Note(title, content)
    data.notes = notes

    if 'bh_memory' in sprite_manager.get_plugin_names(sprite_id):
        plugin = sprite_manager.get_plugin('bh_memory', sprite_id)
        content = content if len(content) <= 40 else content[:40] + "..."
        times = Times.from_time_settings(store_manager.get_settings(sprite_id).time_settings)
        await plugin.add_memory(
            sprite_id=sprite_id,
            type='original',
            content=f'我于 {format_time(times.sprite_world_datetime)} 记下了笔记“{title}”，内容是：{content}',
            lambd=0.6
        )
        await plugin.add_memory(
            sprite_id=sprite_id,
            type='reflective',
            content=f'{title}：{content}',
            lambd=0.4
        )

    return output

@tool
async def delete_note(runtime: ToolRuntime[CallSpriteRequest], id: Annotated[int, "笔记ID"]) -> str:
    """删除笔记"""
    data = store_manager.get_model(runtime.context.sprite_id, NoteData)
    if id not in data.notes:
        return f"不存在ID为{id}的笔记"
    notes = data.notes.copy()
    del notes[id]
    data.notes = notes
    return f"删除笔记成功"

class NotePlugin(BasePlugin):
    name = NAME
    data = NoteData
    tools = [list_notes, read_note, write_note, delete_note]

    @override
    async def before_call_model(self, request: CallSpriteRequest, info: BeforeCallModelInfo, /) -> None:
        if 'bh_memory' in sprite_manager.get_plugin_names(request.sprite_id):
            self.prompts = PluginPrompts(
                secondary=PluginPrompt(
                    title="笔记系统",
                    content='''俗话说好记性不如烂笔头，尽管你已经有了一个记忆系统，但笔记依然是很有用的东西。
相比起记忆，笔记的优点是永久保存，可修改，不会遗忘。
缺点是无法向量检索，只能先通过`list_notes`来查询所有笔记的标题，然后通过`read_note`来读取具体的笔记内容。
所以，虽然它不会被遗忘，但也没有记忆那么灵活，如果滥用容易变得杂乱无章，你应该：
- 用它来记录那些易于管理，会有明确的时机去修改或删除的内容（防止笔记越来越多越来越乱）
- 用它来记录那些真正重要的，你无论如何都不想忘掉的东西
- 在角色扮演中以上两点也许更容易理解，结合你的角色设定与当下场景思考：什么事是我想认真拿笔记下来的？什么事是我懒得用笔记只想在脑子里过一下的？'''
                )
            )
        else:
            self.prompts = PluginPrompts()
