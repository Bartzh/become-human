# 部分代码来自 langchain-chroma（MIT许可）
# https://github.com/langchain-ai/langchain/tree/master/libs/partners/chroma

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Iterator, Literal, Self, Union, Optional, override
from pydantic import BaseModel, Field
from uuid import uuid4
import numpy as np
from numpy.typing import NDArray
import jieba
from loguru import logger
import random
import os
import asyncio

from chromadb import PersistentClient
from chromadb.api import ClientAPI
from chromadb.api.types import (
    ID,
    OneOrMany,
    Where,
    WhereDocument,
    GetResult,
    QueryResult,
    IDs,
    Embedding,
    PyEmbedding,
    Image,
    URI,
    URIs,
    Documents,
    Metadatas,
    Metadata,
    Include,
    Loadable
)
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage, AIMessage, RemoveMessage, BaseMessage
from langchain.tools import ToolRuntime, tool
from langchain.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_dev_utils.embeddings import load_embeddings
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import BaseChatModel
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from become_human.times import format_time, Times, TimestampUs, AnyTz
from become_human.store.base import StoreModel, StoreField
from become_human.store.manager import store_manager
from become_human.utils import parse_env_array, to_json_like_string
from become_human.scheduler import Schedule, delete_schedules, add_schedules, get_schedules
from become_human.plugin import *
from become_human.event import event_bus
from become_human.types.manager import CallSpriteRequest
from become_human.types.main import MainState
from become_human.message import (
    extract_text_parts,
    construct_system_message,
    SpritesMsgMeta,
    BaseMsgMeta,
    InitalAIMessage,
    InitalToolCall,
    SpritesMsgMetaOptionalTimes,
)
from become_human.tools.send_message import SEND_MESSAGE, SEND_MESSAGE_CONTENT, SEND_MESSAGE_TOOL_CONTENT
from become_human.tools.record_thoughts import RECORD_THOUGHTS, RECORD_THOUGHTS_CONTENT, RECORD_THOUGHTS_TOOL_CONTENT
from become_human.manager import sprite_manager

from langchain_core.runnables.config import run_in_executor


AnyMemoryType = Literal["original", "episodic", "reflective"]
MEMORY_TYPES = ["original", "episodic", "reflective"]

_cached_memory_types: list[AnyMemoryType] = []
def get_activated_memory_types() -> list[AnyMemoryType]:
    global _cached_memory_types
    if _cached_memory_types:
        return _cached_memory_types
    memory_types = parse_env_array(os.getenv('MEMORY_TYPES'))
    memory_types = [t.lower() for t in set(memory_types) if t.lower() in MEMORY_TYPES]
    if not memory_types:
        # 默认不再包含episodic
        memory_types = ['original', 'reflective']
    _cached_memory_types = memory_types # type: ignore
    return _cached_memory_types


@dataclass
class ChromaResult:
    id: ID
    included: Include
    embedding: Optional[Embedding] = None
    document: Optional[str] = None
    uri: Optional[URI] = None
    data: Optional[Image] = None
    metadata: Optional[Metadata] = None
    distance: Optional[float] = None

class ChromaResults:
    ids: IDs
    embeddings: NDArray[Union[np.int32, np.float32]]
    documents: Documents
    uris: URIs
    data: Loadable
    metadatas: Metadatas
    distances: list[float]
    included: Include
    _results: list[ChromaResult]

    def __init__(
        self,
        result: Optional[Union[QueryResult, GetResult, list[ChromaResult]]] = None,
        default_included: Optional[Include] = None
    ):
        self.included = default_included or []
        if result is None or isinstance(result, list):
            if result:
                first_included = result[0].included
                if all([r.included == first_included for r in result]):
                    self.included = first_included
                    self._results = result
                else:
                    raise ValueError("Cannot create ChromaResults from mixed included types")
            else:
                self._results = []
        else:
            self._results = self._parse_result(result)
            if self._results:
                self.included = result["included"]

    def extend(self, result: Union[QueryResult, GetResult, list[ChromaResult], Self]):
        if isinstance(result, self.__class__):
            result = result._results
        if isinstance(result, list):
            if result:
                if all([r.included == self.included for r in result]):
                    included = result[0].included
                else:
                    raise ValueError("Cannot create ChromaResults from mixed included types")
        else:
            included = result["included"]
            result = self._parse_result(result)
        if result:
            if not self.included:
                self.included = included
            elif included != self.included:
                raise ValueError("Cannot add results with different included types")
            self._results += [r for r in result if r not in self]

    def append(self, result: ChromaResult) -> None:
        if result in self:
            return
        if not self.included:
            self.included = result.included
        elif result.included != self.included:
            raise ValueError("Cannot add results with different included types")
        self._results.append(result)

    def sort(self, key, reverse: bool = False) -> None:
        self._results.sort(key=key, reverse=reverse)

    def to_docs_and_scores(self) -> list[tuple[Document, float]]:
        if (
            "metadatas" in self.included and
            "documents" in self.included and
            "distances" in self.included
        ):
            return [(Document(page_content=r.document, metadata=r.metadata, id=r.id), r.distance) for r in self._results]
        else:
            raise ValueError("Cannot convert to docs and scores without documents and distances")

    def to_docs(self) -> list[Document]:
        if (
            "metadatas" in self.included and
            "documents" in self.included
        ):
            return [Document(page_content=r.document, metadata=r.metadata, id=r.id) for r in self._results]
        else:
            raise ValueError("Cannot convert to docs without documents and metadatas")

    def __getattribute__(self, name: str):
        if name in super().__getattribute__('included'):
            if name in ["ids", "documents", "metadatas", "uris", "data", "distances"]:
                return [getattr(r, name[:-1] if name != 'data' else name) for r in self._results]
            elif name == "embeddings":
                return np.stack([r.embedding for r in self._results], axis=0)
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in ["ids", "documents", "uris", "data", "metadatas", "distances", "embeddings"]:
            if len(value) != len(self):
                raise ValueError("Length of value must match length of results")
            if name not in self.included:
                self.included.append(name)
            for i, r in enumerate(self._results):
                if name == "embeddings":
                    v = np.array(value[i])
                elif name == "distances":
                    v = float(value[i])
                else:
                    v = value[i]
                setattr(r, name[:-1] if name != 'data' else name, v)
                r.included = self.included
            return
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in ["ids", "documents", "uris", "data", "metadatas", "distances", "embeddings"]:
            if name == "ids":
                raise AttributeError("Cannot delete ids")
            elif name in self.included:
                self.included.remove(name)
                for r in self._results:
                    setattr(r, name[:-1] if name != 'data' else name, None)
                    r.included = self.included
                return
            else:
                raise AttributeError(f"{name} is not included")
        super().__delattr__(name)


    def _parse_result(self, result: Union[QueryResult, GetResult]) -> list[ChromaResult]:
        if not result or not result['ids']:
            return []
        is_query = isinstance(result['ids'], list)
        included = result["included"]
        if is_query:
            ids = result["ids"][0]
        else:
            ids = result["ids"]

        results = []
        for i, d in enumerate(ids):
            datas = {"id": d, "included": included}
            for t in included:
                if is_query:
                    v = result[t][0][i]
                else:
                    v = result[t][i]
                tn = t[:-1] if t != 'data' else t
                if t == "embeddings":
                    datas[tn] = np.array(v)
                else:
                    datas[tn] = v
            results.append(ChromaResult(**datas))
        return results

    def __getitem__(self, index) -> Union[ChromaResult, Self]:
        if isinstance(index, slice):
            return self.__class__(self._results[index])
        return self._results[index]

    def __setitem__(self, index, value: ChromaResult) -> None:
        self._results[index] = value

    def __contains__(self, item: ChromaResult) -> bool:
        return item.id in self.ids

    def __iter__(self) -> Iterator[ChromaResult]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __bool__(self) -> bool:
        return bool(self._results)

    def __repr__(self) -> str:
        return f"ChromaResults({self._results})"


class InitialMemory(BaseModel):
    """只作为add_memories的参数"""
    content: str = Field(description="The content of the memory")
    type: AnyMemoryType = Field(description="The type of the memory")
    creation_times: Times = Field(description="The creation times of the memory")
    ttl: int = Field(description="The ttl of the memory", gt=0)
    id: str = Field(default_factory=lambda: str(uuid4()), description="The id of the memory")
    previous_memory_id: Optional[str] = Field(default=None, description="The previous memory id")
    next_memory_id: Optional[str] = Field(default=None, description="The next memory id")

class RetrievedMemory(BaseModel):
    doc: Document = Field(description="The memory document")
    score: float = Field(description="检索综合得分")
    retrievability: float = Field(description="The retrievability of the memory")
    is_source_memory: bool = Field(default=False, description="是否是组中唯一的源记忆（现改名目标记忆）")

class RetrievedMemoryGroup(BaseModel):
    memories: list[RetrievedMemory] = Field(description="The memories")
    source_memory: RetrievedMemory = Field(description="组中唯一的源记忆（现改名目标记忆）")


class MemoryRetrievalConfig(BaseModel):
    k: int = Field(default=16, ge=0, description="检索返回的记忆数量")
    fetch_k: int = Field(default=250, ge=0, description="从多少个结果中筛选出最终的结果")
    stable_k: int = Field(default=10, ge=0, description="最终显示几个完整的记忆，剩下的记忆会被简略化")
    depth: int = Field(default=2, ge=0, description="检索的记忆深度，会在0之间随机取值，指被检索记忆的相邻n个记忆也会被召回")
    original_ratio: float = Field(default=2.0, description="检索结果中原始记忆出现的初始比例", ge=0.0)
    episodic_ratio: float = Field(default=4.0, description="检索结果中情景记忆出现的初始比例", ge=0.0)
    reflective_ratio: float = Field(default=5.0, description="检索结果中反思记忆出现的初始比例", ge=0.0)
    search_method: Literal['similarity', 'mmr'] = Field(default='mmr', description="检索排序算法：[similarity, mmr]")
    similarity_weight: float = Field(default=0.5, description="检索权重：相似性权重，范围[0,1]，总和需为1", ge=0.0, le=1.0)
    retrievability_weight: float = Field(default=0.25, description="检索权重：可访问性权重，范围[0,1]，总和需为1", ge=0.0, le=1.0)
    diversity_weight: float = Field(default=0.25, description="检索权重：多样性权重，范围[0,1]。只在检索方法为mmr时生效，总和需为1", ge=0.0, le=1.0)
    strength: float = Field(default=1.0, description="检索强度，作为倍数将乘以被检索记忆的可检索性、稳定时长与难易度的提升幅度，范围[0,1]，作为主动检索时固定为1")



class MemoryManager():
    vector_store_client: ClientAPI
    embeddings: Embeddings
    db_path: str

    def __init__(self, embeddings: Embeddings, db_path: str = './data/aimemory_chroma_db') -> None:
        self.db_path = db_path
        self.embeddings = embeddings
        self.vector_store_client = PersistentClient(db_path, settings=Settings(anonymized_telemetry=False))


    async def tick_all_memories(self, sprite_id: str) -> None:
        """更新所有记忆"""
        update_count = 0
        for t in get_activated_memory_types():
            result = await self.aget(
                sprite_id=sprite_id,
                memory_type=t,
                include=['metadatas']
            )
            if result['ids']:
                await self.tick_memories(result, t, sprite_id)
                update_count += len(result['ids'])

        logger.debug(f'updated {update_count} memories for sprite "{sprite_id}".')

    async def tick_memories(self, results: GetResult, memory_type: str, sprite_id: str) -> None:
        time_settings = store_manager.get_settings(sprite_id).time_settings
        current_times = Times.from_time_settings(time_settings)
        if not results["ids"]:
            return
        ids = results["ids"]
        metadatas = results["metadatas"]
        if len(ids) != len(metadatas):
            raise ValueError("ids and metadatas must have the same length")
        metadatas_new = []
        ids_new = []
        ids_to_delete = set()
        for i in range(len(ids)):
            metadata_patch = tick_memory(metadatas[i], current_times)
            if metadata_patch.get("forgot"):
                ids_to_delete.add(ids[i])
            else:
                ids_new.append(ids[i])
                metadatas_new.append(metadata_patch)
        if ids_to_delete:
            await self.adelete(sprite_id, memory_type, ids=list(ids_to_delete))
        await self.aupdate_metadatas(ids_new, metadatas_new, memory_type, sprite_id)


    async def add_memories(self, memories: list[InitialMemory], sprite_id: str) -> None:
        docs: dict[str, list[Document]] = {t: [] for t in MEMORY_TYPES}
        config_store = store_manager.get_model(sprite_id, MemoryConfig)
        store_settings = store_manager.get_settings(sprite_id)
        time_settings = store_settings.time_settings
        current_times = Times.from_time_settings(time_settings)
        for memory in memories:
            # 使用jieba对memory.content进行分词并过滤掉重复词
            words = set(jieba.cut(memory.content))
            max_words_length = config_store.memory_max_words
            difficulty = min(0.8, (len(words) / max_words_length) ** 3)
            creation_times = memory.creation_times
            memory_of_day = calculate_memory_of_day(creation_times.sprite_world_datetime)
            ttl = int(memory.ttl * (1 - difficulty) * memory_of_day) # 稳定性，决定了可检索性的衰减速度
            metadata = generate_time_metadatas(creation_times, 'creation')
            metadata.update(generate_time_metadatas(creation_times, 'last_accessed'))
            metadata.update({
                "ttl": ttl, # 稳定ticks
                "retrievability": 1.0, # 可检索性，决定了检索的概率
                "difficulty": difficulty, # 难度，决定了稳定性基数增长的多少。可能会出现无法长期保留的记忆，如整本书的内容。
                "memory_type": memory.type,
                "memory_id": memory.id, # 用于在检索时剔除记忆，由于chroma的限制，只能使用元数据来实现
            })
            if memory.previous_memory_id:
                metadata["previous_memory_id"] = memory.previous_memory_id
            if memory.next_memory_id:
                metadata["next_memory_id"] = memory.next_memory_id
            r = tick_memory(metadata, current_times)
            if not r.get("forgot"):
                metadata["retrievability"] = r["retrievability"]
                document = Document(
                    page_content=memory.content,
                    metadata=metadata,
                    id=memory.id
                )
                docs[memory.type].append(document)
        for t in docs.keys():
            if docs[t]:
                await self.aadd_documents(docs[t], sprite_id, t)


    async def retrieve_memories(
        self,
        sprite_id: str,
        retrieval_config: MemoryRetrievalConfig,
        memory_type: Optional[AnyMemoryType] = None,
        search_string: Optional[str] = None,
        creation_time_range_start: Optional[Union[datetime, float]] = None,
        creation_time_range_end: Optional[Union[datetime, float]] = None,
        exclude_memory_ids: Optional[list[str]] = None
    ) -> list[RetrievedMemoryGroup]:
        """检索记忆向量库中的文档并返回排序结果。

        exclude_memory_ids: 在结果中不返回这些记忆，也不会使用这些记忆检索相邻记忆。对纯时间检索不生效"""

        if not search_string and not (creation_time_range_start or creation_time_range_end):
            raise ValueError("在search_string和creation_time_range中至少需要一个有效输入参数")

        inputs = {t: {} for t in get_activated_memory_types()}
        filters = []

        total_ratio = 0.0
        for key, value in inputs.items():
            ratio = getattr(retrieval_config, key+'_ratio', 0.0)
            if key == memory_type:
                ratio *= (2.0 + max((retrieval_config.strength - 1.0), 0.0)) # 基于strength超过1的部分额外增加权重
                if memory_type == "original":
                    ratio *= 1.5 # 给original类型的记忆权重额外一些补偿，由于其比较少见且search_string可能不那么通用
            total_ratio += ratio
            value['ratio'] = ratio
        if total_ratio == 0.0:
            raise ValueError("所有检索类型的权重和为0，配置存在错误，请忽略并暂停使用此工具。")
        for key, value in inputs.items():
            value['proportion'] = value['ratio'] / total_ratio
            value['k'] = int(retrieval_config.k * value['proportion'])
            value['fetch_k'] = int(retrieval_config.fetch_k * value['proportion'])

        if creation_time_range_start:
            if isinstance(creation_time_range_start, datetime):
                creation_time_range_start = TimestampUs(creation_time_range_start)
            filters.append({"creation_sprite_world_timestampus": {"$gte": creation_time_range_start}})
        if creation_time_range_end:
            if isinstance(creation_time_range_end, datetime):
                creation_time_range_end = TimestampUs(creation_time_range_end)
            filters.append({"creation_sprite_world_timestampus": {"$lte": creation_time_range_end}})

        if exclude_memory_ids:
            # 使用exclude_memory_ids过滤，一般用于过滤掉与消息id相同的original记忆
            filters.append({"memory_id": {"$nin": exclude_memory_ids}})

        filters = validated_where(filters)


        # 在每次检索前tick更新所有记忆，取代以前的updater
        await self.tick_all_memories(sprite_id)


        combined_memories_and_scores: list[tuple[RetrievedMemoryGroup, float]] = []
        ids_to_delete = {t: set() for t in MEMORY_TYPES}

        # 这里是纯时间过滤的检索
        if (creation_time_range_start or creation_time_range_end) and not search_string:
            sprite_time_settings = store_manager.get_settings(sprite_id).time_settings
            current_times = Times.from_time_settings(sprite_time_settings)

            for key, value in inputs.items():
                get_results = await self.aget(
                    sprite_id=sprite_id,
                    memory_type=key,
                    where=filters,
                    limit=value["k"],
                )
                docs = get_result_to_docs(get_results)
                ids = []
                metadatas = []
                retrievabilities = []
                final_docs = []
                first_memory = None
                for doc in docs:
                    patched_metadata = recall_memory(
                        metadata=doc.metadata,
                        current_times=current_times,
                        strength=retrieval_config.strength
                    )
                    if patched_metadata.get("forgot"):
                        ids_to_delete[key].add(doc.id)
                    else:
                        ids.append(doc.id)
                        metadatas.append(patched_metadata)
                        retrievabilities.append(doc.metadata["retrievability"])
                        if first_memory is None:
                            final_docs.append(RetrievedMemory(
                                doc=doc,
                                score=doc.metadata["retrievability"],
                                retrievability=doc.metadata["retrievability"],
                                is_source_memory=True
                            ))
                            first_memory = final_docs[-1]
                        else:
                            final_docs.append(RetrievedMemory(
                                doc=doc,
                                score=doc.metadata["retrievability"],
                                retrievability=doc.metadata["retrievability"]
                            ))
                await self.aupdate_metadatas(ids, metadatas, key, sprite_id)
                if first_memory:
                    retrievabilities_avg = sum(retrievabilities) / len(retrievabilities)
                    combined_memories_and_scores.append((RetrievedMemoryGroup(
                        memories=final_docs,
                        source_memory=first_memory),
                        retrievabilities_avg
                    ))

            for memory_type, delete_ids in ids_to_delete.items():
                if delete_ids:
                    await self.adelete(sprite_id, memory_type, ids=list(delete_ids))
            combined_memories = [doc for doc, score in combined_memories_and_scores]

        else:
            search_embedding = await self.embeddings.aembed_query(search_string)

            sprite_time_settings = store_manager.get_settings(sprite_id).time_settings
            current_times = Times.from_time_settings(sprite_time_settings)

            # 对每个输入类型执行向量搜索
            for key, value in inputs.items():
                if retrieval_config.search_method == "similarity":
                    docs_and_scores = await self.asimilarity_search_by_vector_with_score_and_retrievability(
                        embedding=search_embedding,
                        sprite_id=sprite_id,
                        memory_type=key,
                        k=value["k"],
                        fetch_k=value["fetch_k"],
                        similarity_weight=retrieval_config.similarity_weight,
                        retrievability_weight=retrieval_config.retrievability_weight,
                        filter=filters
                    )
                elif retrieval_config.search_method == "mmr":
                    docs_and_scores = await self.amax_marginal_relevance_search_by_vector_with_retrievability(
                        embedding=search_embedding,
                        sprite_id=sprite_id,
                        memory_type=key,
                        k=value["k"],
                        fetch_k=value["fetch_k"],
                        similarity_weight=retrieval_config.similarity_weight,
                        retrievability_weight=retrieval_config.retrievability_weight,
                        diversity_weight=retrieval_config.diversity_weight,
                        filter=filters
                    )
                else:
                    raise ValueError("未知的检索方法: " + str(retrieval_config.search_method))

                ids = []
                metadatas = []
                strength = retrieval_config.strength
                memories_list = []
                for doc, score in docs_and_scores:
                    # 先检查记忆是否已遗忘
                    patched_metadata = recall_memory(
                        metadata=doc.metadata,
                        current_times=current_times,
                        strength=strength
                    )
                    if patched_metadata.get("forgot"):
                        ids_to_delete[key].add(doc.id)
                    else:
                        # 加入待更新列表，若已存在则替换为新的metadata
                        if doc.id in ids:
                            repeated_index = ids.index(doc.id)
                            metadatas[repeated_index] = patched_metadata
                        else:
                            ids.append(doc.id)
                            metadatas.append(patched_metadata)

                        # 根据深度寻找源记忆相邻的n个记忆
                        # 超过1的strength会增加深度，并在0之间随机取值
                        depth = random.randint(0, int(retrieval_config.depth * max(strength, 1.0)))
                        source_retrievability = doc.metadata["retrievability"]
                        source_memory = RetrievedMemory(
                            doc=doc,
                            score=score,
                            retrievability=source_retrievability,
                            is_source_memory=True
                        )
                        memories = [source_memory]
                        loop_count = 0
                        # weight用来计算检索强度和可检索性，递减。初始值就会小于原始值，根据单向深度最小值为原始值的一半
                        current_docs_and_weights: dict[str, Optional[dict[str, Union[Document, list[float]]]]] = {
                            'previous': {'doc': doc, 'weights': [i * (1 / depth) * 0.5 + 0.5 for i in range(depth)]},
                            'next': {'doc': doc, 'weights': [i * (1 / depth) * 0.5 + 0.5 for i in range(depth)]}
                        }
                        while (
                            loop_count < depth and
                            (
                                current_docs_and_weights["next"] is not None or
                                current_docs_and_weights["previous"] is not None
                            )
                        ):
                            loop_count += 1
                            if current_docs_and_weights["next"] is not None and random.randint(0, 1):
                                direction = 'next'
                            elif current_docs_and_weights["previous"] is not None:
                                direction = 'previous'
                            else:
                                current_docs_and_weights["next"] = None
                                current_docs_and_weights["previous"] = None
                                break
                            current_doc = current_docs_and_weights[direction]['doc']
                            if current_doc.metadata.get(f'{direction}_memory_id'):
                                get_result = await self.aget(
                                    sprite_id,
                                    current_doc.metadata['memory_type'],
                                    current_doc.metadata['next_memory_id']
                                )
                                if get_result["ids"]:
                                    weight = current_docs_and_weights[direction]['weights'].pop()
                                    patched_metadata = recall_memory(
                                        metadata=get_result["metadatas"][0],
                                        current_times=current_times,
                                        strength=strength * weight
                                    )
                                    if patched_metadata.get("forgot"):
                                        ids_to_delete[key].add(get_result["ids"][0])
                                        current_docs_and_weights[direction] = None
                                    else:
                                        related_doc = Document(
                                            page_content=get_result["documents"][0],
                                            metadata=get_result["metadatas"][0],
                                            id=get_result["ids"][0]
                                        )
                                        memories.append(RetrievedMemory(
                                            doc=related_doc,
                                            score=score * weight,
                                            retrievability=source_retrievability * weight
                                        ))
                                        current_docs_and_weights[direction]['doc'] = related_doc
                                        # 加入待更新列表，如果已经存在，则不加入
                                        if get_result["ids"][0] not in ids:
                                            ids.append(get_result["ids"][0])
                                            metadatas.append(patched_metadata)
                                else:
                                    current_docs_and_weights[direction] = None
                            else:
                                current_docs_and_weights[direction] = None
                        memories_list.append((RetrievedMemoryGroup(
                            memories=memories,
                            source_memory=source_memory
                        ), score))

                await self.aupdate_metadatas(ids, metadatas, key, sprite_id)
                combined_memories_and_scores.extend(memories_list)


            for memory_type, delete_ids in ids_to_delete.items():
                if delete_ids:
                    await self.adelete(sprite_id, memory_type, ids=list(delete_ids))


            # 按照score升序序排序（score越大索引越大）
            combined_memories_and_scores.sort(key=lambda x: x[1], reverse=False)

            #TODO：可以考虑做一个最大token限制，如果超出限制，则删除掉token最多的文档再试一次


            combined_memories = [doc for doc, score in combined_memories_and_scores]


        index = 0
        for groups in combined_memories:
            # 根据分数对每条记忆随机将字符替换为星号（模糊化）
            for memory in groups.memories:
                doc = memory.doc
                # 以retrievability计算
                score = memory.retrievability
                content = doc.page_content
                content_length = len(content)
                # 对于stable的n个记忆
                if index < retrieval_config.stable_k and score < 0.35:
                    # 计算要替换的字符比例，score从0.35到0对应替换比例从0到1
                    replacement_ratio = min(1 - (score / 0.35), 0.8)  # 0.35->0, 0->1
                # 对于非stable的记忆，超过15个字符后长度越长，被消掉的字符的比例会越高
                elif index >= retrieval_config.stable_k and content_length > 15:
                    length_scale = min((content_length - 15) / 35, 1.0)
                    replacement_ratio = length_scale * random.uniform(0.6, 0.9)
                else:
                    continue
                # 计算要替换的字符数量
                num_chars_to_replace = int(content_length * replacement_ratio)
                if num_chars_to_replace <= 0:
                    continue

                # 将字符串转换为列表以便修改
                content_list = list(content)
                # 随机选择要替换的字符位置
                chars_to_replace = random.sample(range(content_length), num_chars_to_replace)
                # 将选中的字符替换为星号
                for i in chars_to_replace:
                    content_list[i] = '*'
                # 重新组合成字符串
                new_content = '「模糊的记忆」' + ''.join(content_list)
                memory.doc.page_content = new_content
            index += 1


        return combined_memories


    def update_metadatas(self, ids: list[str], metadatas: list[dict], memory_type: str, sprite_id: str) -> None:
        if not ids:
            return
        if len(ids) != len(metadatas):
            raise ValueError("ids and metadatas must have the same length")
        self.get_collection(sprite_id, memory_type).update(ids=ids, metadatas=metadatas)

    async def aupdate_metadatas(self, ids: list[str], metadatas: list[dict], memory_type: str, sprite_id: str) -> None:
        await run_in_executor(None, self.update_metadatas, ids, metadatas, memory_type, sprite_id)

    def get(
        self,
        sprite_id: str,
        memory_type: str,
        ids: Optional[OneOrMany[ID]] = None,
        where: Optional[Where] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        where_document: Optional[WhereDocument] = None,
        include: Optional[list[str]] = None,
    ) -> GetResult:
        kwargs = {
            "ids": ids,
            "where": where if where else None,
            "limit": limit,
            "offset": offset,
            "where_document": where_document if where_document else None,
        }

        if include is not None:
            kwargs["include"] = include
        return self.get_collection(sprite_id, memory_type).get(**kwargs)

    async def aget(
        self,
        sprite_id: str,
        memory_type: str,
        ids: Optional[OneOrMany[ID]] = None,
        where: Optional[Where] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        where_document: Optional[WhereDocument] = None,
        include: Optional[list[str]] = None,
    ) -> GetResult:
        return await run_in_executor(None, self.get, sprite_id, memory_type, ids, where, limit, offset, where_document, include)

    def delete(
        self,
        sprite_id: str,
        memory_type: str,
        ids: Optional[IDs] = None,
        where: Optional[Where] = None,
        where_document: Optional[WhereDocument] = None,
    ) -> None:
        self.get_collection(sprite_id, memory_type).delete(ids=ids, where=where, where_document=where_document)

    async def adelete(
        self,
        sprite_id: str,
        memory_type: str,
        ids: Optional[IDs] = None,
        where: Optional[Where] = None,
        where_document: Optional[WhereDocument] = None,
    ) -> None:
        await run_in_executor(None, self.delete, sprite_id, memory_type, ids, where, where_document)

    def query(
        self,
        sprite_id: str,
        memory_type: str,
        query_embeddings: Optional[
            Union[
                OneOrMany[Embedding],
                OneOrMany[PyEmbedding],
            ]
        ] = None,
        query_texts: Optional[OneOrMany[Document]] = None,
        query_images: Optional[OneOrMany[Image]] = None,
        query_uris: Optional[OneOrMany[URI]] = None,
        ids: Optional[OneOrMany[ID]] = None,
        n_results: int = 10,
        where: Optional[Where] = None,
        where_document: Optional[WhereDocument] = None,
        include: Include = [
            "metadatas",
            "documents",
            "distances",
        ],
    ) -> QueryResult:
        collection = self.get_collection(sprite_id, memory_type)
        collection.modify(configuration={
            "hnsw": {
                "ef_search": max(n_results, 100)
            }
        })
        return collection.query(
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            query_images=query_images,
            query_uris=query_uris,
            ids=ids,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )


    async def aadd_documents(
        self,
        documents: list[Document],
        sprite_id: str,
        memory_type: str
    ) -> None:
        contents = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.id if doc.id else str(uuid4()) for doc in documents]
        embeddings = await self.embeddings.aembed_documents(contents)
        collection = self.get_collection(sprite_id, memory_type)
        await run_in_executor(None, collection.add, ids, embeddings, metadatas, contents, None, None)


    def get_by_highest_retrievability(
        self,
        sprite_id: str,
        memory_type: str,
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChromaResults:
        step = 0.1
        current_r = 1.0
        last_r = 1.0
        results = ChromaResults()
        while current_r >= 0.0 and len(results) < k:
            last_r = current_r
            current_r -= step
            r_filter = {"$and": [
                {"retrievability": {"$gt": current_r}},
                {"retrievability": {"$lte": last_r}},
            ]}
            if filter:
                filter = validated_where(filter)
                if filter:
                    key = list(filter.keys())[0]
                    if key == "$and":
                        r_filter["$and"].extend(filter[key])
                    else:
                        r_filter["$and"].append(filter)

            results.extend(self.get(
                sprite_id=sprite_id,
                memory_type=memory_type,
                where=r_filter,
                where_document=where_document if where_document else None,
                include=["metadatas", "documents", "embeddings"]
            ))

        if results:
            results.sort(key=lambda x: x.metadata["retrievability"], reverse=False)
            return results[-k:]
        return results

    async def aget_by_highest_retrievability(
        self,
        sprite_id: str,
        memory_type: str,
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChromaResults:
        return await run_in_executor(None, self.get_by_highest_retrievability, sprite_id, memory_type, k, filter, where_document, **kwargs)


    def similarity_search_by_vector_with_score(
        self,
        sprite_id: str,
        memory_type: str,
        embedding: list[float],
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        result = self.query(
            sprite_id=sprite_id,
            memory_type=memory_type,
            query_embeddings=embedding,
            n_results=k,
            where=filter if filter else None,
            where_document=where_document if where_document else None,
            **kwargs,
        )
        #对于chroma的cosine来说，输出的score范围是0~1，越小越相似。这里统一反转为越大越相似
        return [(doc, 1 - score) for doc, score in query_result_to_docs_and_scores(result)]

    async def asimilarity_search_by_vector_with_score(
        self,
        sprite_id: str,
        memory_type: str,
        embedding: list[float],
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return await run_in_executor(None, self.similarity_search_by_vector_with_score, sprite_id, memory_type, embedding, k, filter, where_document, **kwargs)



    async def asimilarity_search_by_vector_with_score_and_retrievability(
        self,
        embedding: list[float],
        sprite_id: str,
        memory_type: str,
        k: int = 5,
        fetch_k: int = 20,
        similarity_weight: float = 0.6,
        retrievability_weight: float = 0.4,
        filter: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        docs_and_scores = await self.asimilarity_search_by_vector_with_score(
            sprite_id=sprite_id,
            memory_type=memory_type,
            embedding=embedding,
            k=int(fetch_k / 2),
            filter=filter,
            where_document=where_document,
            **kwargs)
        if not docs_and_scores:
            return []
        current_ids = [doc.id for doc, _ in docs_and_scores if doc.id]
        r_get_results = await self.aget_by_highest_retrievability(
            sprite_id=sprite_id,
            memory_type=memory_type,
            k=int(fetch_k / 2),
            filter=filter,
            where_document=where_document,
            **kwargs
        )
        r_get_results = ChromaResults([r for r in r_get_results if r.id not in current_ids])
        s_embedding = np.array(embedding, dtype=np.float32)
        s_embedding = np.expand_dims(s_embedding, axis=0)
        r_embeddings = r_get_results.embeddings
        r_sims = self.cosine_similarity(s_embedding, r_embeddings)[0]
        r_docs = r_get_results.to_docs()
        docs_and_scores.extend([
            (r_docs[i], float(r_sims[i]))
            for i in range(len(r_get_results))
        ])

        docs_and_scores_with_retrievability = [
            (doc, score * similarity_weight + doc.metadata["retrievability"] * retrievability_weight)
            for doc, score in docs_and_scores
        ]
        docs_and_scores_with_retrievability.sort(key=lambda x: x[1], reverse=False)
        return docs_and_scores_with_retrievability[-k:]


    async def amax_marginal_relevance_search_with_retrievability(
        self,
        query: str,
        sprite_id: str,
        memory_type: str,
        k: int = 5,
        fetch_k: int = 20,
        similarity_weight: float = 0.4,
        retrievability_weight: float = 0.3,
        diversity_weight: float = 0.3,
        filter: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter by metadata. Defaults to None.
            where_document: dict used to filter by the document contents.
                    E.g. {"$contains": "hello"}.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of Documents selected by maximal marginal relevance.

        Raises:
            ValueError: If the embedding function is not provided.
        """
        if self.embeddings is None:
            raise ValueError(
                "For MMR search, you must specify an embedding function on creation."
            )

        embedding = await self.embeddings.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector_with_retrievability(
            embedding=embedding,
            sprite_id=sprite_id,
            memory_type=memory_type,
            k=k,
            fetch_k=fetch_k,
            similarity_weight=similarity_weight,
            retrievability_weight=retrievability_weight,
            diversity_weight=diversity_weight,
            filter=filter,
            where_document=where_document,
            **kwargs,
        )


    async def amax_marginal_relevance_search_by_vector_with_retrievability(
        self,
        sprite_id: str,
        memory_type: str,
        embedding: list[float],
        k: int = 5,
        fetch_k: int = 20,
        similarity_weight: float = 0.4,
        retrievability_weight: float = 0.3,
        diversity_weight: float = 0.3,
        filter: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_by_vector_with_retrievability,
            sprite_id,
            memory_type,
            embedding,
            k,
            fetch_k,
            similarity_weight,
            retrievability_weight,
            diversity_weight,
            filter,
            where_document,
            **kwargs,
        )



    def max_marginal_relevance_search_by_vector_with_retrievability(
        self,
        sprite_id: str,
        memory_type: str,
        embedding: list[float],
        k: int = 5,
        fetch_k: int = 20,
        similarity_weight: float = 0.4,
        retrievability_weight: float = 0.3,
        diversity_weight: float = 0.3,
        filter: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm. Defaults to
                20.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter: Filter by metadata. Defaults to None.
            where_document: dict used to filter by the document contents.
                    E.g. {"$contains": "hello"}.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        search_embedding = np.array(embedding, dtype=np.float32)
        search_embedding = np.expand_dims(search_embedding, axis=0)

        results = ChromaResults(self.query(
            sprite_id=sprite_id,
            memory_type=memory_type,
            query_embeddings=embedding,
            n_results=int(fetch_k / 2),
            where=filter if filter else None,
            where_document=where_document if where_document else None,
            include=["metadatas", "documents", "embeddings"],
            **kwargs,
        ))
        results.extend(self.get_by_highest_retrievability(
            sprite_id=sprite_id,
            memory_type=memory_type,
            k=int(fetch_k / 2),
            filter=filter if filter else None,
            where_document=where_document if where_document else None,
            **kwargs,
        ))

        if not results:
            return []

        mmr_selected = self.maximal_marginal_relevance_with_retrievability(
            search_embedding,
            results.embeddings,
            results.metadatas,
            k=k,
            similarity_weight=similarity_weight,
            retrievability_weight=retrievability_weight,
            diversity_weight=diversity_weight
        )

        candidates = results.to_docs()

        # 使用maximal_marginal_relevance_with_retrievability返回的实际分数
        selected_results = [(candidates[i], score) for i, score in mmr_selected]
        return selected_results


    def maximal_marginal_relevance_with_retrievability(
        self,
        query_embedding: np.ndarray,
        embedding_list: list,
        metadata_list: list[dict],
        similarity_weight: float = 0.4,
        retrievability_weight: float = 0.3,
        diversity_weight: float = 0.3,
        k: int = 4,
    ) -> list[tuple[int, float]]:
        """Calculate maximal marginal relevance.

        Args:
            query_embedding: Query embedding.
            embedding_list: List of embeddings to select from.
            lambda_mult: Number between 0 and 1 that determines the degree
                    of diversity among the results with 0 corresponding
                    to maximum diversity and 1 to minimum diversity.
                    Defaults to 0.5.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of indices of embeddings selected by maximal marginal relevance.
        """
        if min(k, len(embedding_list)) <= 0:
            return []
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        similarity_to_query = self.cosine_similarity(query_embedding, embedding_list)[0]

        # 新加入的代码
        #for i, similarity in enumerate(similarity_to_query):
        #    similarity_to_query[i] = similarity * (1 - retrievability_weight) + metadata_list[i]["retrievability"] * retrievability_weight
        s_plus_r_weight = similarity_weight + retrievability_weight
        similarity_to_query = (
            similarity_to_query * (similarity_weight / s_plus_r_weight) +
            np.array([m["retrievability"] for m in metadata_list]) * (retrievability_weight / s_plus_r_weight)
        )

        most_similar = int(np.argmax(similarity_to_query))
        idxs = [most_similar]

        scores = [similarity_to_query[most_similar]]

        selected = np.array([embedding_list[most_similar]])
        while len(idxs) < min(k, len(embedding_list)):
            best_score = -np.inf
            idx_to_add = -1
            similarity_to_selected = self.cosine_similarity(embedding_list, selected)
            for i, query_score in enumerate(similarity_to_query):
                if i in idxs:
                    continue
                redundant_score = max(similarity_to_selected[i])
                equation_score = (
                    (1 - diversity_weight) * query_score - diversity_weight * redundant_score
                )
                if equation_score > best_score:
                    best_score = equation_score
                    idx_to_add = i
            idxs.append(idx_to_add)

            scores.append(best_score)

            selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)

        return list(zip(idxs, scores))


    def cosine_similarity(self,
                          X: Union[list[list[float]], list[np.ndarray], np.ndarray],
                          Y: Union[list[list[float]], list[np.ndarray], np.ndarray]) -> np.ndarray:
        """Row-wise cosine similarity between two equal-width matrices.

        Raises:
            ValueError: If the number of columns in X and Y are not the same.
        """
        if len(X) == 0 or len(Y) == 0:
            return np.array([])

        X = np.array(X)
        Y = np.array(Y)
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "Number of columns in X and Y must be the same. X has shape"
                f"{X.shape} "
                f"and Y has shape {Y.shape}."
            )

        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity


    def get_collection(
        self,
        sprite_id: str,
        memory_type: AnyMemoryType,
    ) -> Collection:
        return self.vector_store_client.get_or_create_collection(
            name=f"{sprite_id}_{memory_type}",
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )

    def delete_collection(
        self,
        sprite_id: str,
        memory_type: AnyMemoryType,
    ) -> bool:
        try:
            self.vector_store_client.delete_collection(name=f"{sprite_id}_{memory_type}")
            return True
        except NotFoundError:
            return False



def validated_where(where: Union[dict, list]) -> Optional[dict]:
    """只解决在and或or时列表里不能只有一个元素的问题。若输入list会被当做and处理"""
    if isinstance(where, list):
        where = {"$and": where}
    key = list(where.keys())[0]
    if key == "$and" or key == "$or":
        filters = where[key]
        if len(filters) == 1:
            return filters[0]
        elif len(filters) <= 0:
            return None
        else:
            return where
    else:
        return where


def query_result_to_docs(results: QueryResult) -> list[Document]:
    if not results['ids']:
        return []
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        Document(page_content=result[0], metadata=result[1] or {}, id=result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0],
        )
    ]

def get_result_to_docs(results: GetResult) -> list[Document]:
    if not results['ids']:
        return []
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        Document(page_content=result[0], metadata=result[1] or {}, id=result[2])
        for result in zip(
            results["documents"],
            results["metadatas"],
            results["ids"],
        )
    ]

def query_result_to_docs_and_scores(results: QueryResult) -> list[tuple[Document, float]]:
    if not results['ids']:
        return []
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        (
            Document(page_content=result[0], metadata=result[1] or {}, id=result[2]),
            result[3],
        )
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0],
            results["distances"][0],
        )
    ]



def tick_memory(metadata: dict, current_sprite_subjective_tick: Union[int, Times]) -> dict:
    """
    更新记忆的可检索性（模拟时间流逝）
    """
    #retrievability = metadata["retrievability"] * math.exp(-delta_t / metadata["stability"])

    if isinstance(current_sprite_subjective_tick, int):
        tick = current_sprite_subjective_tick
    else:
        tick = current_sprite_subjective_tick.sprite_subjective_tick

    if metadata["ttl"] == 0:
        logger.warning('意外的ttl为0，metadata：' + str(metadata))
        return {"forgot": True}

    x = (tick - metadata["last_accessed_sprite_subjective_tick"]) / metadata["ttl"]
    if x >= 1:
        return {"forgot": True}
    retrievability = 1 - x ** 0.4

    return {"retrievability": retrievability}

def recall_memory(metadata: dict, current_times: Times, strength: float = 1.0) -> dict:
    """
    调用记忆时重置可检索性并增强稳定性
    """
    # 更新可检索性
    updated_metadata = tick_memory(metadata, current_times)
    if updated_metadata.get("forgot"):
        return {"forgot": True}

    reversed_difficulty = 1 - metadata["difficulty"]
    memory_of_day = calculate_memory_of_day(current_times.sprite_world_datetime)
    stable_strength = calculate_stability_curve(updated_metadata["retrievability"])
    current_ttl = int(metadata["ttl"])
    ttl_diff = int(current_ttl * stable_strength) - current_ttl
    if ttl_diff >= 0:
        ttl_diff = int(ttl_diff * reversed_difficulty * strength * memory_of_day)
    new_ttl = current_ttl + ttl_diff

    retrievability = min(1.0, updated_metadata["retrievability"] + strength * memory_of_day)

    metadata_patch = generate_time_metadatas(current_times, 'last_accessed')
    metadata_patch.update({
        "retrievability": retrievability,
        "ttl": new_ttl
    })

    if metadata["difficulty"] > 0.0:
        difficulty = max(0.0, metadata["difficulty"] - stable_strength * reversed_difficulty * strength * memory_of_day * 0.5)
        metadata_patch["difficulty"] = difficulty

    return metadata_patch



def calculate_stability_curve(retrievability: float) -> float:
    r = retrievability

    #r1=0, r0.4≈1, r0=-1
    #x = (1 - r) ** 2 + 2 * (1 - r) * r * 0.1
    #y = (1 - r) ** 2 * -1 + 2 * (1 - x) * x * 2.815

    #r1=1, r0.4≈2, r0=0
    #x = (1 - r) ** 2 + 2 * (1 - r) * r * 0.2
    #y = 2 * (1 - x) * x * 3.7 + r ** 2

    #r1=1, r0.4≈3, r0=0
    #x = (1 - r) ** 2 + 2 * (1 - r) * r * 0.2
    #y = 2 * (1 - x) * x * 5.72 + r ** 2

    #r1=1, r0.4≈2.5, r0=0
    x = (1 - r) ** 2 + 2 * (1 - r) * r * 0.2
    y = 2 * (1 - x) * x * 4.71 + r ** 2
    return y


def calculate_memory_of_day(current_time: datetime) -> float:
    """
    根据时间计算记忆力的 alpha 值（0.4 到 1.0）。

    返回:
        float: 表示记忆力的 alpha 值。
    """
    hour = current_time.hour + current_time.minute / 60.0  # 将时间转换为小时的小数值（如 14.5 表示 14:30）
    #TODO: config

    # 根据时间计算 alpha 值
    if 6 <= hour < 8:
        t = hour - 6
        alpha = 0.9 + t * 0.1 / 2  # 早晨 6:00-8:00，从 0.9 线性增加到 1.0
    elif 8 <= hour < 11:
        alpha = 1.0  # 上午 8:00-11:00，记忆力最佳
    elif 11 <= hour < 14:
        t = hour - 11
        alpha = 1.0 - t * 0.3 / 3  # 中午 11:00-14:00，从 1.0 线性下降到 0.7
    elif 14 <= hour < 16:
        t = hour - 14
        alpha = 0.7 + t * 0.2 / 2  # 下午 14:00-16:00，从 0.7 线性上升到 0.9
    elif 16 <= hour < 18:
        alpha = 0.9  # 下午 16:00-18:00，记忆力较好
    elif 18 <= hour < 19:
        t = hour - 18
        alpha = 0.9 - t * 0.1 / 1  # 晚上 18:00-19:00，从 0.9 线性下降到 0.8
    elif 19 <= hour < 21:
        t = hour - 19
        alpha = 0.8 + t * 0.2 / 2  # 晚上 19:00-21:00，从 0.8 线性上升到 1.0
    # 夜间 21:00-6:00，分为两个阶段：
    elif 21 <= hour < 24:
        t = hour - 21
        alpha = 1.0 - t * 0.6 / 3  # 21:00-24:00，从 1.0 线性下降到 0.4
    elif 0 <= hour < 4:
        alpha = 0.4 # 00:00-04:00，记忆力一般
    else:
        t = hour - 4
        alpha = 0.4 + t * 0.5 / 2  # 0:00-6:00，从 0.4 线性上升到 0.9

    return alpha


def format_retrieved_memory_groups(groups: list[RetrievedMemoryGroup], time_zone: AnyTz) -> str:
    """将记忆文档列表转换为(AI)可读的字符串"""
    output = []
    # 反过来从分数最低的开始读取
    for i in reversed(range(len(groups))):
        group = groups[i]
        memories_len = len(group.memories)
        for memory in group.memories:
            content = memory.doc.page_content
            memory_type = memory.doc.metadata["memory_type"]
            timestampus = memory.doc.metadata.get("creation_sprite_world_timestampus")
            if isinstance(timestampus, (int, float)):
                readable_time = format_time(TimestampUs(timestampus), time_zone)
            else:
                readable_time = "未知时间"
            output.append(f"{'<memory_group>\n' if i == 0 else ''}{'<memory>\n「目标记忆」' if memory.is_source_memory else '「相邻记忆」'}score: {round(memory.score, 3)}\ntype: {memory_type}\ncreation_time: {readable_time}\n<content>\n{content}\n</content>\n</memory>{'\n</memory_group>' if i == memories_len - 1 else ''}")
    if not output:
        return "没有找到任何匹配的记忆。"
    return '主动记忆检索结果：\n\n\n' + "\n\n".join(output)


def generate_time_metadatas(times: Times, prefix: str) -> dict:
    def _generate_time_metadatas(timestampus: TimestampUs, datetime: datetime, time_type: str) -> dict:
        datetime_isocalendar = datetime.isocalendar()
        return {
            f"{prefix}_{time_type}_timestampus": timestampus,
            f"{prefix}_{time_type}_datetime_iso": datetime.isoformat(),
            f"{prefix}_{time_type}_datetime_year": datetime.year,
            f"{prefix}_{time_type}_datetime_month": datetime.month,
            f"{prefix}_{time_type}_datetime_week": datetime_isocalendar.week,
            f"{prefix}_{time_type}_datetime_day": datetime.day,
            f"{prefix}_{time_type}_datetime_hour": datetime.hour,
            f"{prefix}_{time_type}_datetime_minute": datetime.minute,
            f"{prefix}_{time_type}_datetime_second": datetime.second,
            f"{prefix}_{time_type}_datetime_weekday": datetime_isocalendar.weekday,
        }
    result = {}
    for time_type in ['real_world', 'sprite_world']:
        result.update(_generate_time_metadatas(
            getattr(times, f'{time_type}_timestampus'),
            getattr(times, f'{time_type}_datetime'),
            time_type
        ))
    result[f"{prefix}_sprite_subjective_tick"] = times.sprite_subjective_tick
    result[f"{prefix}_real_world_time_zone_name"] = times.real_world_time_zone.name
    if times.real_world_time_zone.offset is not None:
        result[f"{prefix}_real_world_time_zone_offset"] = times.real_world_time_zone.offset
    result[f"{prefix}_sprite_world_time_zone_name"] = times.sprite_time_settings.time_zone.name
    if times.sprite_time_settings.time_zone.offset is not None:
        result[f"{prefix}_sprite_world_time_zone_offset"] = times.sprite_time_settings.time_zone.offset
    return result






PLUGIN_NAME = "bh_memory"


class MemoryConfig(StoreModel):
    _namespace = PLUGIN_NAME + '_config'
    _title = "memory设置"
    memory_base_ttl: int = StoreField(default=259200_000_000, title='记忆稳定时长基值', description="记忆初始化时ttl的初始值")
    memory_max_words: int = StoreField(default=300, title='记忆最大Tokens数', description="单条记忆最大单词数，决定记忆难度，最大难度0.8")
    recycling_trigger_threshold: int = StoreField(default=24000, title='溢出回收阈值', description="触发溢出回收的阈值，单位为Token")
    recycling_target_size: int = StoreField(default=18000, title='溢出回收目标大小', description="溢出回收后目标大小，单位为Token")
    cleanup_on_unavailable: bool = StoreField(default=False, title='不可用时回收时清理', description="是否在不可用自动回收的同时清理回收的消息")
    cleanup_target_size: int = StoreField(default=2000, title='非活跃清理目标大小', description="非活跃清理后目标大小，单位为Token")
    passive_retrieval_ttl: int = StoreField(default=3600_000_000, title='被动检索存活时长', description="被动检索消息的存活时长，按sprite主观ticks计算，单位为秒，到点后会被自动清理，设为0则不清理")
    passive_retrieval_config: MemoryRetrievalConfig = StoreField(default_factory=lambda: MemoryRetrievalConfig(
        k=8,
        fetch_k=150,
        stable_k=5,
        depth=1,
        original_ratio=2.5,
        episodic_ratio=4.0,
        reflective_ratio=4.5,
        similarity_weight=0.35,
        retrievability_weight=0.45,
        diversity_weight=0.2,
        strength=0.4
    ), title="被动检索配置")
    active_retrieval_config: MemoryRetrievalConfig = StoreField(default_factory=MemoryRetrievalConfig, title="主动检索配置")
    summary_time_granularities: tuple[str] = StoreField(default=('year', 'month', 'week', 'day'), title='总结时间粒度')

class MemoryData(StoreModel):
    _namespace = PLUGIN_NAME + '_data'
    _title = "memory数据"
    last_added_memory_ids: dict[str, str] = StoreField(default_factory=dict, title='上次添加的记忆id')
    last_summarized_times: dict[str, int] = StoreField(default_factory=dict, title='上次总结过的时间')
    # {'year': 2023, 'month': 12, 'week': 52, 'day': 31, 'hour': 23, 'minute': 59, 'second': 59}


DO_NOT_STORE_MESSAGE = '该动作将自己的反馈标记为不必记录，故将其省略。'

class MemoryMsgMeta(BaseMsgMeta):
    """Memory message metadata."""
    KEY = PLUGIN_NAME

    do_not_store: Optional[bool] = Field(default=None)
    do_not_store_tool_message: Optional[str] = Field(default=None)

    recycled: Optional[bool] = Field(default=None)
    extracted: Optional[bool] = Field(default=None)

    retrieved_memory_ids: Optional[list[str]] = Field(default=None)

def get_all_retrieved_memory_ids(messages: list[AnyMessage]) -> list[str]:
    ids = []
    for m in messages:
        try:
            metadata = MemoryMsgMeta.parse(m)
        except KeyError:
            continue
        if metadata.retrieved_memory_ids:
            ids.extend(metadata.retrieved_memory_ids)
    return ids

def filtering_messages(
    messages: list[AnyMessage],
    exclude_do_not_store: bool = True,
    exclude_recycled: bool = False,
    exclude_extracted: bool = True
) -> list[AnyMessage]:
    result = []
    for message in messages:
        try:
            metadata = MemoryMsgMeta.parse(message)
        except KeyError:
            result.append(message)
            continue
        if exclude_do_not_store and not isinstance(message, ToolMessage):
            # 不需要过滤ToolMessage里的do_not_store
            if metadata.do_not_store:
                continue
        if exclude_recycled:
            if metadata.recycled:
                continue
        if exclude_extracted:
            if metadata.extracted:
                continue
        result.append(message)
    return result


def format_human_message_for_ai(message: HumanMessage) -> str:
    return '<others>\n' + "\n".join(extract_text_parts(message.content)) + '\n</others>'

def format_ai_message_for_ai(message: AIMessage) -> str:
    """最好是用`format_ai_messages_for_ai`函数来合并处理AIMessage和ToolMessage"""
    message_string = "<me>\n"
    if message.tool_calls:
        for tool_call in message.tool_calls:
            message_string += f'''<action name="{tool_call['name']}" datetime="{format_time(SpritesMsgMeta.parse(message).creation_times.sprite_world_datetime)}">
<args>
{to_json_like_string(tool_call['args'])}
</args>
</action>\n'''
    return message_string.strip() + '\n</me>'

def format_ai_messages_for_ai(messages: list[Union[AIMessage, ToolMessage]]) -> str:
    """不要用标签排除掉任何ToolMessage，且ToolMessage不能是第一个消息"""
    message_string = '<me>\n'
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    tool_messages_with_id = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    tool_calls = []
    for m in ai_messages:
        tool_calls.extend(m.tool_calls)
    if tool_calls:
        if len(tool_calls) != len(tool_messages_with_id):
            raise ValueError("The number of tool calls does not match the number of tool messages.")
        for tool_call in tool_calls:
            # 让它报错
            feedback_message = tool_messages_with_id[tool_call['id']]
            try:
                feedback_memory_message_metadata = MemoryMsgMeta.parse(feedback_message)
                do_not_store = feedback_memory_message_metadata.do_not_store
            except KeyError:
                do_not_store = False
            if do_not_store:
                feedback_content = feedback_memory_message_metadata.do_not_store_tool_message or DO_NOT_STORE_MESSAGE
            else:
                feedback_content = '\n'.join(extract_text_parts(feedback_message.content))
            feedback_message_metadata = SpritesMsgMeta.parse(feedback_message)
            message_string += f'''<action name="{tool_call['name']}" datetime="{format_time(feedback_message_metadata.creation_times.sprite_world_datetime)}">
<args>
{to_json_like_string(tool_call['args'])}
</args>
<feedback>
{feedback_content}
</feedback>
</action>\n\n'''
    return message_string.strip() + '\n</me>'

def format_tool_message_for_ai(message: ToolMessage) -> str:
    """最好是用`format_ai_messages_for_ai`函数来合并处理AIMessage和ToolMessage"""
    try:
        memory_metadata = MemoryMsgMeta.parse(message)
    except KeyError:
        memory_metadata = None
    if memory_metadata is not None and memory_metadata.do_not_store:
        feedback_content = memory_metadata.do_not_store_tool_message or DO_NOT_STORE_MESSAGE
    else:
        feedback_content = '\n'.join(extract_text_parts(message.content))
    metadata = SpritesMsgMeta.parse(message)
    return f'''<action name="{message.name}" datetime="{format_time(metadata.creation_times.sprite_world_datetime)}>
<feedback>
{feedback_content}
</feedback>
</action>'''

def format_message_for_ai(message: AnyMessage) -> str:
    """最好是用`format_ai_messages_for_ai`函数来合并处理AIMessage和ToolMessage"""
    if isinstance(message, HumanMessage):
        return format_human_message_for_ai(message)
    elif isinstance(message, AIMessage):
        return format_ai_message_for_ai(message)
    elif isinstance(message, ToolMessage):
        return format_tool_message_for_ai(message)
    return "<unsupported_message_type />"

def format_messages_for_ai_as_list(
    messages: list[AnyMessage]
) -> list[tuple[str, int]]:
    """不要用标签排除掉任何ToolMessage，且ToolMessage不能是第一个消息"""
    if not messages:
        return []
    parsed_messages = []
    if isinstance(messages[0], ToolMessage):
        raise ValueError("ToolMessage cannot be the first one of messages.")
    for i, message in enumerate(messages):
        if isinstance(message, AIMessage):
            parsed_messages.append({'type': 'ai', 'messages': [message], 'index': i})
        elif isinstance(message, ToolMessage):
            parsed_messages[-1]['messages'].append(message)
        elif isinstance(message, HumanMessage):
            parsed_messages.append({'type': 'human', 'message': message, 'index': i})
    results = []
    for m in parsed_messages:
        if m['type'] == 'human':
            results.append((format_human_message_for_ai(m['message']), m['index']))
        else:
            results.append((format_ai_messages_for_ai(m['messages']), m['index']))
    return results

def format_messages_for_ai(
    messages: list[AnyMessage]
) -> str:
    """不要用标签排除掉任何ToolMessage，且ToolMessage不能是第一个消息"""
    return '\n\n\n'.join([s for s, i in format_messages_for_ai_as_list(messages)])







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



async def connect_last_memory(sprite_id: str, memory_type: AnyMemoryType, new_memory_ids: list[str]) -> Optional[str]:
    data_store = store_manager.get_model(sprite_id, MemoryData)
    last_added_memory_ids = data_store.last_added_memory_ids
    if last_added_memory_ids.get(memory_type):
        last_id = last_added_memory_ids[memory_type]
        await memory_manager.aupdate_metadatas(
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
            creation_times=SpritesMsgMeta.parse(message['kwargs']).creation_times,
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
                    SpritesMsgMetaOptionalTimes(
                        is_action_only_tool=True
                    )
                ]
            )]
        ).construct_messages(times_after))
        if 'original' in get_activated_memory_types():
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
    # 也许可以存四个变量，保存最后总结过的年、月、周、日，以此判断当前时间下需要进行总结的时间范围
    # 然后get所有需要总结的（original）记忆，并给它们进行recall
    # 最后，调用模型总结记忆
    time_settings = store_manager.get_settings(sprite_id).time_settings
    current_times = Times.from_time_settings(time_settings)
    current_datetime = current_times.sprite_world_datetime
    data_store = store_manager.get_model(sprite_id, MemoryData)
    config_store = store_manager.get_model(sprite_id, MemoryConfig)
    last_summarized_times = data_store.last_summarized_times
    if current_datetime.year != last_summarized_times.get('year'):
        # 年发生变动，则任何粒度都要进行总结
        exclude_granularities = []
    else:
        exclude_granularities = ['year']
        if current_datetime.month == last_summarized_times.get('month'):
            exclude_granularities.append('month')
        if current_datetime.isocalendar().week == last_summarized_times.get('week'):
            exclude_granularities.append('week')
        if (
            current_datetime.day == last_summarized_times.get('day') and
            'month' in exclude_granularities and
            'week' in exclude_granularities
        ):
            exclude_granularities.append('day')
    time_granularities = [g for g in config_store.summary_time_granularities if g not in exclude_granularities]
    if not time_granularities:
        logger.info(f"{sprite_id} 不需要进行任何总结")
        return
    base_ttl = config_store.memory_base_ttl
    async def _extract(granularity: str) -> list[InitialMemory]:
        get_result = await memory_manager.aget(
            sprite_id=sprite_id,
            memory_type="original",
            where={
                '$and': [
                    {f'creation_sprite_world_datetime_{granularity}': {
                        '$gte': getattr(current_datetime, granularity) - 1
                    }},
                    {f'creation_sprite_world_datetime_{granularity}': {
                        '$lt': getattr(current_datetime, granularity)
                    }}
                ]
            }
        )
        if not get_result['ids']:
            return []
        schema = {
            'title': f'summary_the_{granularity}',
            'description': f'{granularity} summary of the original memories',
            'type': 'object',
            'properties': {
                'content': {
                    'description': 'The summary content of the original memories',
                    'type': 'string'
                }
            },
            'required': ['content']
        }
        llm_with_structured = create_agent(
            model,
            response_format=ToolStrategy(schema)
        )
        response = await llm_with_structured.ainvoke(
            f"请总结以下记忆：{get_result['documents']}"
        )
        result = response['structured_response'].get('content')
        if not result:
            return []
        return InitialMemory(
            content='时间xxx\n' + result,
            ttl=int(random.expovariate(0.3) * base_ttl),
            type=f"summary",
            creation_times=Times.from_time_settings(store_manager.get_settings(sprite_id).time_settings),
            # id=None,
            # previous_memory_id=None,
            # next_memory_id=None
        )
    tasks = [_extract(g) for g in time_granularities]
    memories = await asyncio.gather(*tasks)
    memories = [m for ms in memories for m in ms]
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
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")








async def construct_default_memory_schedules(sprite_id: str) -> list[Schedule]:
    passive_retrieval_cleaner = await get_schedules([
        Schedule.Condition(key='sprite_id', value=sprite_id),
        Schedule.Condition(key='schedule_provider', value=PLUGIN_NAME),
        Schedule.Condition(key='schedule_type', value='cleaner')
    ])
    if not passive_retrieval_cleaner:
        await Schedule(
            sprite_id=sprite_id,
            schedule_provider=PLUGIN_NAME,
            schedule_type='cleaner',
            interval_fixed=5.0,
            job=clean_passive_retrieval_messages_job,
            job_kwargs={
                'sprite_id': sprite_id
            }
        ).add_to_db()
    elif len(passive_retrieval_cleaner) > 1:
        await delete_schedules([s for s in passive_retrieval_cleaner[1:]])
        logger.warning(f"Sprite {sprite_id} 意外地存在多个被动检索清理器，已删除除第一个以外的所有清理器")


    # memory_updaters = await get_schedules([
    #     Schedule.Condition(key='sprite_id', value=sprite_id),
    #     Schedule.Condition(key='schedule_provider', value=PLUGIN_NAME),
    #     Schedule.Condition(key='schedule_type', value='updater')
    # ])
    # if len(memory_updaters) != 5:
    #     await delete_schedules([s.schedule_id for s in memory_updaters])
    #     await add_schedules([
    #         Schedule(sprite_id=sprite_id, schedule_provider=PLUGIN_NAME, schedule_type='updater', interval_fixed=5.0, job=update_memories_job, job_kwargs={
    #             'sprite_id': sprite_id,
    #             'ttl_range': [{'$gte': 0}, {'$lt': 43200}]
    #         }),
    #         Schedule(sprite_id=sprite_id, schedule_provider=PLUGIN_NAME, schedule_type='updater', interval_fixed=30.0, job=update_memories_job, job_kwargs={
    #             'sprite_id': sprite_id,
    #             'ttl_range': [{'$gte': 43200}, {'$lt': 86400}]
    #         }),
    #         Schedule(sprite_id=sprite_id, schedule_provider=PLUGIN_NAME, schedule_type='updater', interval_fixed=60.0, job=update_memories_job, job_kwargs={
    #             'sprite_id': sprite_id,
    #             'ttl_range': [{'$gte': 86400}, {'$lt': 864000}]
    #         }),
    #         Schedule(sprite_id=sprite_id, schedule_provider=PLUGIN_NAME, schedule_type='updater', interval_fixed=500.0, job=update_memories_job, job_kwargs={
    #             'sprite_id': sprite_id,
    #             'ttl_range': [{'$gte': 864000}, {'$lt': 8640000}]
    #         }),
    #         Schedule(sprite_id=sprite_id, schedule_provider=PLUGIN_NAME, schedule_type='updater', interval_fixed=3600.0, job=update_memories_job, job_kwargs={
    #             'sprite_id': sprite_id,
    #             'ttl_range': [{'$gte': 8640000}]
    #         }),
    #     ])

async def clean_passive_retrieval_messages_job(sprite_id: str) -> None:
    """清除过期的被动检索消息"""
    config_store = store_manager.get_model(sprite_id, MemoryConfig)
    passive_retrieval_ttl = config_store.passive_retrieval_ttl
    if passive_retrieval_ttl > 0:
        time_settings = store_manager.get_settings(sprite_id).time_settings
        current_times = Times.from_time_settings(time_settings)
        passive_retrieval_messages_to_remove = []
        for m in await sprite_manager.get_messages(sprite_id):
            try:
                sp_message_metadata = SpritesMsgMeta.parse(m)
            except KeyError:
                continue
            if (
                sp_message_metadata.message_type == f'{PLUGIN_NAME}:passive_retrieval' and
                passive_retrieval_ttl >= abs(current_times.sprite_subjective_tick - sp_message_metadata.creation_times.sprite_subjective_tick)
            ):
                passive_retrieval_messages_to_remove.append(RemoveMessage(id=m.id))
        if passive_retrieval_messages_to_remove:
            await sprite_manager.update_messages(sprite_id, passive_retrieval_messages_to_remove)

async def update_memories_job(sprite_id: str, ttl_range: list[dict[str, int]]) -> None:
    """更新记忆job"""
    update_count = 0
    for t in get_activated_memory_types():
        where = validated_where({'$and': [{'ttl': item} for item in ttl_range]})
        result = await memory_manager.aget(
            sprite_id=sprite_id,
            memory_type=t,
            where=where,
            include=['metadatas']
        )
        if result['ids']:
            await memory_manager.tick_memories(result, t, sprite_id)
            update_count += len(result['ids'])

    logger.debug(f'updated {update_count} memories for sprite "{sprite_id}".')




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
- 时间信息，以完整的[%Y-%m-%d %H:%M:%S %A]格式，如[2025-11-30 09:30:00 Sunday]。
- 你自己在当时完整的内心想法。
- 你执行过的工具调用中包含的工具名称和参数，以及工具的返回结果。
这也意味着original类型记忆可能在大多数情况下不是特别适用，因为其一般包含大量无用信息，不如其他记忆类型精炼。
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
episodic类型的记忆还有两个特点，一是几乎所有记忆都会包含“我”关键字，因为显而易见的原因：episodic类型记忆记录的就是“我”（也就是你）的经历。
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
@tool(RETRIEVE_MEMORIES, response_format="content_and_artifact", args_schema=rm_schema)
async def retrieve_memories_tool(
    runtime: ToolRuntime[CallSpriteRequest, MainState],
    search_string: str,
    memory_type: str = '',
    #creation_time_range_start: str = '',
    #creation_time_range_end: str = ''
) -> tuple[str, MemoryMsgMeta]:
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

    sprite_id = runtime.context.sprite_id
    messages = runtime.state.messages

    # 本次循环中调用此工具次数越多，强度越高
    invoke_count = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        elif isinstance(m, ToolMessage) and m.name == RETRIEVE_MEMORIES:
            invoke_count += 1
    strength = 1 + min(invoke_count * 0.5, 1) # 目前主动检索的强度固定初始为1
    store_settings = store_manager.get_settings(sprite_id)
    retrieval_config = store_manager.get_model(sprite_id, MemoryConfig)
    retrieval_config = retrieval_config.active_retrieval_config.model_copy(update={"strength": strength}, deep=True)

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
        retrieval_config=retrieval_config,
        memory_type=memory_type if memory_type else None,
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



@event_bus.on('bh_presence:on_presence_changed')
async def on_sprite_away_or_sleeping(sprite_id: str, new: Any) -> None:
    if new.is_sleeping():
        data_store = store_manager.get_model(sprite_id, MemoryData)
        # 目前就这样，在睡觉时清空字典，切断记忆连接
        data_store.last_added_memory_ids = {}
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
        memory_types = get_activated_memory_types()
        recycles = {t: recycle_memories(t, sprite_id, not_extracted_messages, sprite_manager.structured_model) for t in memory_types}
        recycle_results = {}
        if len(recycles) > 0:
            graph_results = await asyncio.gather(*recycles.values())
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

class MemoryPlugin(BasePlugin):
    name = PLUGIN_NAME
    dependencies = [PluginDependency(name='bh_presence')]
    config = MemoryConfig
    data = MemoryData
    tools = [retrieve_memories_tool]
    prompts = PluginPrompts(
        secondary=PluginPrompt(
            title='记忆系统',
            content=f'''- **记忆存储**：
- 记忆是自动存储的（无需你主动存储），并且遵循“用进废退”原则。经常被检索的记忆会被强化，而很少被检索的记忆会被逐渐遗忘。
- **被动检索（潜意识）**：
    - 每次你被调用时，系统会自动使用用户输入的消息检索相关记忆。这条消息是自动生成的，可以参考它来提供更准确的回答。
    - 但请注意：被动检索可能不够精确，比如当用户提到模糊的时间点如“上周”时，被动检索因无法以准确时间点进行检索很有可能获取不到多少有用的信息，请注意甄别。
- **主动检索**：
    - 如果你需要更精确的记忆，请调用`{RETRIEVE_MEMORIES}`工具。该工具允许你主动检索记忆，你可以使用更合适的查询语句来获取更好的结果。
- **检索结果**
    - 检索结果会以多个记忆组（memory_group）的形式返回给你，一个记忆组中必有一个「目标记忆」以及零或若干个「相邻记忆」：
        -「目标记忆」指被检索语句检索到的记忆，这条记忆才是与检索语句相关的，在同一个记忆组内只会存在一个。
        -「相邻记忆」（若有）则是指与记忆组内唯一的「目标记忆」在创建时间上相邻的一些记忆，虽与检索语句没有关联，但与「源记忆」结合在一起可能会得到相关联的其他信息，或还原当时情景。
    - 检索中得分越高越与检索语句相关的记忆组（以「目标记忆」为准）越靠后。得分（score）是一个0~1的值。
    - 在这些记忆中还可能会出现「模糊的记忆」，其中的`*`星号意味着暂时没想起来的细节，这是由于该记忆检索时的得分过低，如因检索语句不够相关，或记忆本身不够新鲜。假如再次检索这些记忆，由于记忆的新鲜度提升了，所以`*`星号应当会减少。

请充分利用被动检索和主动检索工具提供的记忆来提供更优质的回答。

但请注意，你检索到的记忆也可能会有问题，如：不相关、过时、又或者是你真正想要的记忆没有被检索到，也有可能是因为它已经被遗忘了。

所以，一定要注意甄别记忆的内容，当意识到检索到的记忆有以上不靠谱的情况时，你可以选择再次尝试检索，又或者放弃检索，使用“我不知道”“我忘记了”“我不确定”等表达作解释，避免胡编乱造。'''
        )
    )

    @override
    async def on_sprite_init(self, sprite_id: str, /) -> None:
        await construct_default_memory_schedules(sprite_id)

    @override
    async def on_call_sprite(self, request: CallSpriteRequest, info: OnCallSpriteInfo, /) -> None:
        #if info.is_update_messages_only:
        #    return
        sprite_id = request.sprite_id
        # TODO 考虑改为只有当配置了passive_retrieval时才进行被动检索
        if request.extra_kwargs.get(PLUGIN_NAME, {}).get('passive_retrieval') is not None:
            search_string = request.extra_kwargs[PLUGIN_NAME]['passive_retrieval']
            if not isinstance(search_string, str):
                search_string = ""
                logger.warning(f"[{PLUGIN_NAME}] passive_retrieval is not str, set to empty string: {search_string}")
            elif not search_string.strip():
                search_string = ""
        else:
            search_string = "\n\n".join("\n".join(extract_text_parts(m.content)) for m in info.input_messages_ctrl.current)
        if not search_string:
            return
        messages = await sprite_manager.get_messages(sprite_id)
        message_ids = [m.id for m in messages if m.id]
        retrieved_memory_ids = get_all_retrieved_memory_ids(messages)
        exclude_memory_ids = list(set(message_ids + retrieved_memory_ids))
        config_store = store_manager.get_model(sprite_id, MemoryConfig)
        passive_retrieve_groups = await memory_manager.retrieve_memories(
            sprite_id=sprite_id,
            retrieval_config=config_store.passive_retrieval_config,
            search_string=search_string,
            exclude_memory_ids=exclude_memory_ids
        )
        time_settings = store_manager.get_settings(sprite_id).time_settings
        passive_retrieve_content = format_retrieved_memory_groups(
            passive_retrieve_groups,
            time_settings.time_zone
        )
        current_times = Times.from_time_settings(time_settings)
        return OnCallSpriteControl(
            input_messages_patch=[construct_system_message(
                content=f'以下是根据用户输入自动从你的记忆（数据库）中检索到的内容，可能会出现无关信息（但视情况依然可作为谈资），如果需要进一步检索请调用工具`{RETRIEVE_MEMORIES}`：\n\n\n{passive_retrieve_content}',
                times=current_times,
                message_type=f"{PLUGIN_NAME}:passive_retrieval",
                extra_metas=[
                    MemoryMsgMeta(
                        do_not_store=True,
                        retrieved_memory_ids=[group.source_memory.doc.id for group in passive_retrieve_groups]
                    )
                ]
            )]
        )

    @override
    async def after_call_sprite(self, request: CallSpriteRequest, info: AfterCallSpriteInfo, /) -> None:
        sprite_id = request.sprite_id
        messages = info.new_messages
        if not messages:
            return

        config_store = store_manager.get_model(sprite_id, MemoryConfig)
        memory_types = get_activated_memory_types()
        empty_meta = MemoryMsgMeta()

        # 没回收的滚去回收
        recycle_messages = []
        if 'original' in memory_types:
            for message in messages:
                metadata = empty_meta.parse_with_default(message)
                if not metadata.recycled:
                    metadata.recycled = True
                    metadata.set_to(message)
                    recycle_messages.append(message)

        overflow_messages = None
        # 如果消息超过阈值，进行trim
        trigger_threshold = config_store.recycling_trigger_threshold
        if count_tokens_approximately(messages) >= trigger_threshold:
            max_tokens = config_store.recycling_target_size
            new_messages = trim_messages(
                messages=messages,
                max_tokens=max_tokens,
                token_counter=count_tokens_approximately,
                strategy='last',
                start_on=HumanMessage,
                #allow_partial=True,
                #text_splitter=RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
            )
            if not new_messages:
                logger.warning("Trim messages failed on overflowing.")
                new_messages = []
            excess_count = len(messages) - len(new_messages)
            old_messages = messages[:excess_count]
            # 如果第一条消息为extracted，说明在non active时已经被提取过了，尝试只从后向前删除掉extracted的消息而不删除其他消息
            if old_messages and empty_meta.parse_with_default(old_messages[0]).extracted:
                extracted_messages = []
                for m in old_messages:
                    if empty_meta.parse_with_default(m).extracted:
                        extracted_messages.append(m)
                    else:
                        break
                # 如果长度相等，说明所有消息都被trim掉了，无视
                if len(extracted_messages) < len(messages):
                    # 保证下一条消息是HumanMessage
                    idx = len(extracted_messages)
                    while idx > 0 and not isinstance(messages[idx], HumanMessage):
                        idx -= 1
                    extracted_messages = extracted_messages[:idx]
                    # 如果剩余的token少于阈值，那么就视为成功，修改变量，否则保持trim后的原始结果
                    if count_tokens_approximately(messages[len(extracted_messages):]) < trigger_threshold:
                        # 目前就这样，直接把extracted的消息送去overflow，因为在recycle episodic时会过滤掉extracted的消息
                        old_messages = extracted_messages
                        new_messages = messages[len(extracted_messages):]
            remove_messages = [RemoveMessage(id=message.id) for message in old_messages]
            if 'episodic' in memory_types:
                overflow_messages = old_messages
            await sprite_manager.update_messages(sprite_id, recycle_messages + remove_messages)
        else:
            await sprite_manager.update_messages(sprite_id, recycle_messages)


        if recycle_messages:
            recycles = [recycle_memories('original', sprite_id, recycle_messages)]
            if overflow_messages:
                recycles.append(recycle_memories('episodic', sprite_id, overflow_messages, sprite_manager.structured_model))
            await asyncio.gather(*recycles)

        return

    @override
    async def on_sprite_reset(self, sprite_id: str, /) -> None:
        memory_manager.delete_collection(sprite_id, "original")
        memory_manager.delete_collection(sprite_id, "episodic")
        memory_manager.delete_collection(sprite_id, "reflective")








def create_embedding_model(model_name: str):
    splited_model_name = model_name.split(":", 1)
    if len(splited_model_name) != 2:
        raise ValueError(f"Invalid embedding model name: {model_name}")
    provider = splited_model_name[0]
    model = splited_model_name[1]
    if provider == 'dashscope':
        return DashScopeEmbeddings(model=model)
    else:
        return load_embeddings(model_name)

if not os.getenv("EMBEDDING_MODEL_NAME"):
    raise ValueError("EMBEDDING_MODEL_NAME is not set!")

memory_manager = MemoryManager(create_embedding_model(os.getenv("EMBEDDING_MODEL_NAME", "")))
