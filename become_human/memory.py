import chromadb
from chromadb.api.types import ID, OneOrMany, Where, WhereDocument, GetResult, QueryResult, IDs, Embedding, PyEmbedding, Image, URI, Include
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from langchain.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from datetime import datetime, timezone, timedelta
from typing import Any, Literal, Union, Optional
from pydantic import BaseModel, Field
from uuid import uuid4
import numpy as np
import jieba
from warnings import warn
import random
import os

from become_human.time import format_time, utcnow, real_time_to_agent_time, now_agent_seconds, datetime_to_seconds, now_agent_time, AnyTz
from become_human.store_settings import MemoryRetrievalConfig
from become_human.store_manager import store_manager
from become_human.utils import parse_env_array

from langchain_core.runnables.config import run_in_executor


AnyMemoryType = Literal["original", "episodic", "reflective"]
MEMORY_TYPES = set(["original", "episodic", "reflective"])

_cached_memory_types: list[AnyMemoryType] = []
def get_activated_memory_types() -> list[AnyMemoryType]:
    global _cached_memory_types
    if _cached_memory_types:
        return _cached_memory_types
    memory_types = parse_env_array(os.getenv('MEMORY_TYPES'))
    memory_types = [t.lower() for t in set(memory_types) if t.lower() in MEMORY_TYPES]
    if not memory_types:
        memory_types = ['original', 'episodic', 'reflective']
    _cached_memory_types = memory_types
    return _cached_memory_types

class InitialMemory(BaseModel):
    """只作为add_memories的参数"""
    content: str = Field(description="The content of the memory")
    type: AnyMemoryType = Field(description="The type of the memory")
    #creation_time_seconds: float = Field(description="The creation time seconds of the memory")
    creation_agent_datetime: datetime = Field(description="The creation agent datetime of the memory")
    stable_time: float = Field(description="The stable time of the memory", gt=0.0)
    id: str = Field(default_factory=lambda: str(uuid4()), description="The id of the memory")
    previous_memory_id: Optional[str] = Field(default=None, description="The previous memory id")
    next_memory_id: Optional[str] = Field(default=None, description="The next memory id")

class RetrievedMemory(BaseModel):
    doc: Document = Field(description="The memory document")
    retrievability: float = Field(description="The retrievability of the memory")
    is_source_memory: bool = Field(default=False, description="The source memory of memories")

class RetrievedMemoryGroup(BaseModel):
    memories: list[RetrievedMemory] = Field(description="The memories")
    source_memory: RetrievedMemory = Field(description="The source memory")

class MemoryManager():
    vector_store_client: chromadb.ClientAPI
    embeddings: Embeddings
    db_path: str
    timer_db_path: str

    def __init__(self, embeddings: Embeddings, db_path: str = './data/aimemory_chroma_db') -> None:
        self.db_path = db_path
        self.timer_db_path = f'{db_path}/memory_manager_timers.sqlite'
        self.embeddings = embeddings
        self.vector_store_client = chromadb.PersistentClient(db_path, settings=Settings(anonymized_telemetry=False))


    async def update_timers(self, agent_id: str):
        agent_model = await store_manager.get_agent(agent_id)

        update_count = 0
        timers = agent_model.timers

        settings = agent_model.settings
        types = get_activated_memory_types()
        time_settings = settings.main.time_settings
        current_time = utcnow()
        current_agent_time = real_time_to_agent_time(current_time, time_settings)
        new_timers = []
        for timer in timers.memory_update_timers:
            if timer.is_agent_time:
                time = current_agent_time
            else:
                time = current_time
            next_timer, triggered = timer.calculate_next_timer(time)
            if triggered:
                for t in types:
                    where = validated_where({'$and': [{'stable_time': item} for item in timer.stable_time_range]})
                    result = await self.aget(
                        agent_id=agent_id,
                        memory_type=t,
                        where=where,
                        include=['metadatas']
                    )
                    if result['ids']:
                        await self.update_memories(result, t, agent_id)
                        update_count += len(result['ids'])
            if next_timer is not None:
                new_timers.append(next_timer)
        timers.memory_update_timers = new_timers

        #print(f'updated {update_count} memories for agent "{agent_id}".')


    def update(self, metadata: dict, current_agent_time: Union[float, datetime]) -> dict:
        """
        更新记忆的可检索性（模拟时间流逝）
        """
        #retrievability = metadata["retrievability"] * math.exp(-delta_t / metadata["stability"])

        if isinstance(current_agent_time, datetime):
            current_agent_time_seconds = datetime_to_seconds(current_agent_time)
        else:
            current_agent_time_seconds = current_agent_time

        if metadata["stable_time"] == 0.0:
            print('意外的stable_time为0，metadata：' + str(metadata))
            return {"forgot": True}

        x = (current_agent_time_seconds - metadata["last_accessed_agent_time_seconds"]) / metadata["stable_time"]
        if x >= 1:
            return {"forgot": True}
        retrievability = 1 - x ** 0.4

        return {"retrievability": retrievability}

    def recall(self, metadata: dict, current_agent_time: datetime, strength: float = 1.0) -> dict:
        """
        调用记忆时重置可检索性并增强稳定性
        """
        # 更新可检索性
        current_agent_time_seconds = datetime_to_seconds(current_agent_time)
        updated_metadata = self.update(metadata, current_agent_time_seconds)
        if updated_metadata.get("forgot"):
            return {"forgot": True}

        datetime_alpha = calculate_memory_datetime_alpha(current_agent_time)
        stable_strength = calculate_stability_curve(updated_metadata["retrievability"])
        stable_time_diff = metadata["stable_time"] * stable_strength - metadata["stable_time"]
        if stable_time_diff >= 0:
            stable_time_diff = stable_time_diff * metadata["difficulty"] * strength * datetime_alpha
        stable_time = metadata["stable_time"] + stable_time_diff

        retrievability = min(1.0, updated_metadata["retrievability"] + strength * datetime_alpha)

        metadata_patch = {
            "last_accessed_agent_time_seconds": current_agent_time_seconds,
            "retrievability": retrievability,
            "stable_time": stable_time
        }

        if metadata["difficulty"] < 1.0:
            difficulty = min(1.0, metadata["difficulty"] + stable_strength * metadata["difficulty"] * strength * datetime_alpha * 0.5)
            metadata_patch["difficulty"] = difficulty

        return metadata_patch

    async def update_memories(self, results: GetResult, memory_type: str, agent_id: str) -> None:
        time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
        current_agent_time_seconds = now_agent_seconds(time_settings)
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
            metadata_patch = self.update(metadatas[i], current_agent_time_seconds)
            if metadata_patch.get("forgot"):
                ids_to_delete.add(ids[i])
            else:
                ids_new.append(ids[i])
                metadatas_new.append(metadata_patch)
        if ids_to_delete:
            await self.adelete(agent_id, memory_type, ids=list(ids_to_delete))
        await self.aupdate_metadatas(ids_new, metadatas_new, memory_type, agent_id)


    async def add_memories(self, memories: list[InitialMemory], agent_id: str) -> None:
        docs: dict[str, list[Document]] = {t: [] for t in MEMORY_TYPES}
        time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
        current_agent_time_seconds = now_agent_seconds(time_settings)
        for memory in memories:
            # 使用jieba对memory.content进行分词并过滤掉重复词
            words = set(jieba.cut(memory.content))
            max_words_length = 200
            difficulty = 1 - min(1.0, (len(words) / max_words_length) ** 3)
            creation_agent_datetime = memory.creation_agent_datetime
            datetime_alpha = calculate_memory_datetime_alpha(creation_agent_datetime)
            stable_time = memory.stable_time * difficulty * datetime_alpha # 稳定性，决定了可检索性的衰减速度
            creation_agent_time_seconds = datetime_to_seconds(creation_agent_datetime)
            creation_agent_datetime_isocalendar = creation_agent_datetime.isocalendar()
            metadata = {
                "creation_agent_time_seconds": creation_agent_time_seconds,
                "creation_agent_time_year": creation_agent_datetime.year,
                "creation_agent_time_month": creation_agent_datetime.month,
                "creation_agent_time_week": creation_agent_datetime_isocalendar.week,
                "creation_agent_time_day": creation_agent_datetime.day,
                "creation_agent_time_hour": creation_agent_datetime.hour,
                "creation_agent_time_minute": creation_agent_datetime.minute,
                "creation_agent_time_second": creation_agent_datetime.second,
                "creation_agent_time_weekday": creation_agent_datetime_isocalendar.weekday,
                "stable_time": stable_time, # 稳定时长，单位为秒
                "retrievability": 1.0, # 可检索性，决定了检索的概率
                "difficulty": difficulty, # 难度，决定了稳定性基数增长的多少。可能会出现无法长期保留的记忆，如整本书的内容。
                "last_accessed_agent_time_seconds": creation_agent_time_seconds,
                "memory_type": memory.type,
                "memory_id": memory.id, # 用于在检索时剔除记忆，由于chroma的限制，只能使用元数据来实现
            }
            if memory.previous_memory_id:
                metadata["previous_memory_id"] = memory.previous_memory_id
            if memory.next_memory_id:
                metadata["next_memory_id"] = memory.next_memory_id
            r = self.update(metadata, current_agent_time_seconds)
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
                await self.aadd_documents(docs[t], agent_id, t)


    async def retrieve_memories(
        self,
        agent_id: str,
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
                creation_time_range_start = datetime_to_seconds(creation_time_range_start)
            filters.append({"creation_agent_time_seconds": {"$gte": creation_time_range_start}})
        if creation_time_range_end:
            if isinstance(creation_time_range_end, datetime):
                creation_time_range_end = datetime_to_seconds(creation_time_range_end)
            filters.append({"creation_agent_time_seconds": {"$lte": creation_time_range_end}})

        if exclude_memory_ids:
            # 使用exclude_memory_ids过滤，一般用于过滤掉与消息id相同的original记忆
            filters.append({"memory_id": {"$nin": exclude_memory_ids}})

        filters = validated_where(filters)


        combined_memories_and_scores: list[tuple[RetrievedMemoryGroup, float]] = []
        ids_to_delete = {t: set() for t in MEMORY_TYPES}

        # 这里是纯时间过滤的检索
        if (creation_time_range_start or creation_time_range_end) and not search_string:
            agent_time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
            current_agent_time = now_agent_time(agent_time_settings)

            for key, value in inputs:
                get_results = await self.aget(
                    agent_id=agent_id,
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
                    patched_metadata = self.recall(
                        metadata=doc.metadata,
                        current_agent_time=current_agent_time,
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
                                retrievability=doc.metadata["retrievability"],
                                is_source_memory=True
                            ))
                            first_memory = final_docs[-1]
                        else:
                            final_docs.append(RetrievedMemory(
                                doc=doc,
                                retrievability=doc.metadata["retrievability"]
                            ))
                await self.aupdate_metadatas(ids, metadatas, key, agent_id)
                if first_memory:
                    retrievabilities_avg = sum(retrievabilities) / len(retrievabilities)
                    combined_memories_and_scores.append((RetrievedMemoryGroup(
                        memories=final_docs,
                        source_memory=first_memory),
                        retrievabilities_avg
                    ))

            for memory_type, delete_ids in ids_to_delete.items():
                if delete_ids:
                    await self.adelete(agent_id, memory_type, ids=list(delete_ids))
            combined_memories = [doc for doc, score in combined_memories_and_scores]

        else:
            search_embedding = await self.embeddings.aembed_query(search_string)

            agent_time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
            current_agent_time = now_agent_time(agent_time_settings)

            # 对每个输入类型执行向量搜索
            for key, value in inputs.items():
                if retrieval_config.search_method == "similarity":
                    docs_and_scores = await self.asimilarity_search_by_vector_with_score_and_retrievability(
                        embedding=search_embedding,
                        agent_id=agent_id,
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
                        agent_id=agent_id,
                        memory_type=key,
                        k=value["k"],
                        fetch_k=value["fetch_k"],
                        similarity_weight=retrieval_config.similarity_weight,
                        retrievability_weight=retrieval_config.retrievability_weight,
                        diversity_weight=retrieval_config.diversity_weight,
                        filter=filters
                    )
                else:
                    raise ValueError("未知的检索方法: "+str(retrieval_config.search_method))

                ids = []
                metadatas = []
                strength = retrieval_config.strength
                memories_list = []
                for doc, score in docs_and_scores:
                    # 先检查记忆是否已遗忘
                    patched_metadata = self.recall(
                        metadata=doc.metadata,
                        current_agent_time=current_agent_time,
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
                                    agent_id,
                                    current_doc.metadata['memory_type'],
                                    current_doc.metadata['next_memory_id']
                                )
                                if get_result["ids"]:
                                    weight = current_docs_and_weights[direction]['weights'].pop()
                                    patched_metadata = self.recall(
                                        metadata=get_result["metadatas"][0],
                                        current_agent_time=current_agent_time,
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

                await self.aupdate_metadatas(ids, metadatas, key, agent_id)
                combined_memories_and_scores.extend(memories_list)


            for memory_type, delete_ids in ids_to_delete.items():
                if delete_ids:
                    await self.adelete(agent_id, memory_type, ids=list(delete_ids))


            # 按照score降序排序（score越大索引越小）
            combined_memories_and_scores.sort(key=lambda x: x[1], reverse=True)

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


    def update_metadatas(self, ids: list[str], metadatas: list[dict], memory_type: str, agent_id: str) -> None:
        if not ids:
            return
        if len(ids) != len(metadatas):
            raise ValueError("ids and metadatas must have the same length")
        self.get_collection(agent_id, memory_type).update(ids=ids, metadatas=metadatas)

    async def aupdate_metadatas(self, ids: list[str], metadatas: list[dict], memory_type: str, agent_id: str) -> None:
        await run_in_executor(None, self.update_metadatas, ids, metadatas, memory_type, agent_id)

    def get(
        self,
        agent_id: str,
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
        return self.get_collection(agent_id, memory_type).get(**kwargs)

    async def aget(
        self,
        agent_id: str,
        memory_type: str,
        ids: Optional[OneOrMany[ID]] = None,
        where: Optional[Where] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        where_document: Optional[WhereDocument] = None,
        include: Optional[list[str]] = None,
    ) -> GetResult:
        return await run_in_executor(None, self.get, agent_id, memory_type, ids, where, limit, offset, where_document, include)

    def delete(
        self,
        agent_id: str,
        memory_type: str,
        ids: Optional[IDs] = None,
        where: Optional[Where] = None,
        where_document: Optional[WhereDocument] = None,
    ) -> None:
        self.get_collection(agent_id, memory_type).delete(ids=ids, where=where, where_document=where_document)

    async def adelete(
        self,
        agent_id: str,
        memory_type: str,
        ids: Optional[IDs] = None,
        where: Optional[Where] = None,
        where_document: Optional[WhereDocument] = None,
    ) -> None:
        await run_in_executor(None, self.delete, agent_id, memory_type, ids, where, where_document)

    def query(
        self,
        agent_id: str,
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
        collection = self.get_collection(agent_id, memory_type)
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
        agent_id: str,
        memory_type: str
    ) -> None:
        contents = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.id if doc.id else str(uuid4()) for doc in documents]
        embeddings = await self.embeddings.aembed_documents(contents)
        collection = self.get_collection(agent_id, memory_type)
        await run_in_executor(None, collection.add, ids, embeddings, metadatas, contents, None, None)


    def similarity_search_by_vector_with_score(
        self,
        agent_id: str,
        memory_type: str,
        embedding: list[float],
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        result = self.query(
            agent_id=agent_id,
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
        agent_id: str,
        memory_type: str,
        embedding: list[float],
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return await run_in_executor(None, self.similarity_search_by_vector_with_score, agent_id, memory_type, embedding, k, filter, where_document, **kwargs)



    async def asimilarity_search_by_vector_with_score_and_retrievability(
        self,
        embedding: list[float],
        agent_id: str,
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
            agent_id=agent_id,
            memory_type=memory_type,
            embedding=embedding,
            k=fetch_k,
            filter=filter,
            where_document=where_document,
            **kwargs)
        if not docs_and_scores:
            return []
        docs_and_scores_with_retrievability = [(doc, score * similarity_weight + doc.metadata.get("retrievability", 0) * retrievability_weight) for doc, score in docs_and_scores]
        docs_and_scores_with_retrievability.sort(key=lambda x: x[1], reverse=True)
        return docs_and_scores_with_retrievability[-k:]


    async def amax_marginal_relevance_search_with_retrievability(
        self,
        query: str,
        agent_id: str,
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
            agent_id=agent_id,
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
        agent_id: str,
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
            agent_id,
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
        agent_id: str,
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
        results = self.query(
            agent_id=agent_id,
            memory_type=memory_type,
            query_embeddings=embedding,
            n_results=fetch_k,
            where=filter if filter else None,
            where_document=where_document if where_document else None,
            include=["metadatas", "documents", "distances", "embeddings"],
            **kwargs,
        )
        if not results["ids"]:
            return []
        mmr_selected = self.maximal_marginal_relevance_with_retrievability(
            np.array(embedding, dtype=np.float32),
            results["embeddings"][0],
            results["metadatas"][0],
            k=k,
            similarity_weight=similarity_weight,
            retrievability_weight=retrievability_weight,
            diversity_weight=diversity_weight
        )

        candidates = query_result_to_docs(results)

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
            np.array([m.get("retrievability", 0.0) for m in metadata_list]) * (retrievability_weight / s_plus_r_weight)
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
        agent_id: str,
        memory_type: AnyMemoryType,
    ) -> Collection:
        return self.vector_store_client.get_or_create_collection(
            name=f"{agent_id}_{memory_type}",
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )

    def delete_collection(
        self,
        agent_id: str,
        memory_type: AnyMemoryType,
    ) -> bool:
        try:
            self.vector_store_client.delete_collection(name=f"{agent_id}_{memory_type}")
            return True
        except ValueError:
            return False

def validated_where(where: Union[dict, list]) -> Optional[dict]:
    '''只解决在and或or时列表里不能只有一个元素的问题。若输入list会被当做and处理'''
    if isinstance(where, list):
        where = {"$and": where}
    key = list(where.keys())[0]
    if (key == "$and" or key == "$or"):
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


def calculate_memory_datetime_alpha(current_time: datetime) -> float:
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


def parse_retrieved_memory_groups(groups: list[RetrievedMemoryGroup], time_zone: AnyTz) -> str:
    """将记忆文档列表转换为(AI)可读的字符串"""
    output = []
    # 反过来从分数最低的开始读取
    for i in reversed(range(len(groups))):
        group = groups[i]
        memories_len = len(group.memories)
        for memory in group.memories:
            content = memory.doc.page_content
            memory_type = memory.doc.metadata["memory_type"]
            time_seconds = memory.doc.metadata.get("creation_agent_time_seconds")
            if isinstance(time_seconds, (float, int)):
                readable_time = format_time(time_seconds, time_zone)
            else:
                readable_time = "未知时间"
            # TODO: 太啰嗦，待优化
            output.append(f"{'<记忆组>\n' if i == 0 else ''}{'<记忆>\n「源记忆」' if memory.is_source_memory else '「相邻记忆」'}记忆类型：{memory_type}\n记忆创建时间: {readable_time}\n<记忆内容>\n{content}\n</记忆内容>\n</记忆>{'\n</记忆组>' if i == memories_len - 1 else ''}")
    if not output:
        return "没有找到任何匹配的记忆。"
    # TODO: 也很啰嗦
    return '''记忆检索结果说明：\n1. 检索中得分越高的记忆越靠后。
2.「源记忆」指被检索到的记忆，这条记忆才是与检索语句相关的；「相邻记忆」（若有）则是指「源记忆」在创建时间上相邻的一些记忆，虽与检索语句没有关联，但与「源记忆」结合在一起可能会得到相关联的其他信息，或还原当时情景。
3. 在这些记忆中还可能会出现「模糊的记忆」，其中的`*`星号意味着暂时没想起来的细节，这是由于该记忆检索时的得分过低，如检索语句不够相关，或记忆本身不够新鲜。如果再次检索这些记忆，由于记忆的新鲜度提升了，所以`*`星号应当会减少。
\n\n''' + "\n\n".join(output)


memory_manager = MemoryManager(DashScopeEmbeddings(model="text-embedding-v4"))
