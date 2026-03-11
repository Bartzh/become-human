from typing import Annotated, Optional
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from become_human.store.base import StoreModel, StoreField
from become_human.store.manager import store_manager
from become_human.manager import CallSpriteRequest
from become_human.plugin import *

NAME = 'note'

@dataclass
class Note:
    title: str
    content: str

class NoteData(StoreModel):
    _namespace = NAME + '_data'
    notes: dict[int, Note] = StoreField(default_factory=dict)
    next_id: int = StoreField(default=0)


@tool
async def list_notes(runtime: ToolRuntime[CallSpriteRequest]) -> str:
    """列出所有笔记（的标题）"""
    notes = store_manager.get_model(runtime.context.sprite_id, NoteData).notes
    if not notes:
        return "暂无任何笔记"
    return "\n".join([f"{note_id}. {note.title}" for note_id, note in notes.items()])

@tool
async def read_note(runtime: ToolRuntime[CallSpriteRequest], id: Annotated[int, "笔记ID"]) -> str:
    """读取指定笔记"""
    notes = store_manager.get_model(runtime.context.sprite_id, NoteData).notes
    if not notes:
        return "暂无任何笔记"
    note = notes.get(id)
    if note is None:
        return f"不存在ID为{id}的笔记"
    return note.content

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
    data = store_manager.get_model(runtime.context.sprite_id, NoteData)
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
            content = f"覆盖笔记成功"
    else:
        id = data.next_id
        data.next_id += 1
        content = "新增笔记成功"
    notes = data.notes.copy()
    notes[id] = Note(title, content)
    data.notes = notes
    return content

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
