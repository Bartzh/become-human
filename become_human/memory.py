import chromadb
from chromadb.api.types import ID, OneOrMany, Where, WhereDocument
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
import aiosqlite
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from datetime import datetime, timezone
from typing import Any, Literal, Union, Optional
from pydantic import BaseModel, Field
from uuid import uuid4
import numpy as np
import json
import jieba

from become_human.time import seconds_to_datetime, AgentTimeSettings, get_agent_time_zone, now_seconds

from langchain_core.runnables.config import run_in_executor



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

    @classmethod
    async def create(cls, embeddings: Embeddings, db_path: str = './data/aimemory_chroma_db'):
        instance = cls(embeddings, db_path)
        await instance.init_db()
        return instance


    async def init_db(self):
        async with aiosqlite.connect(self.timer_db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS update_timers (
                    thread_id TEXT PRIMARY KEY,
                    timer_data TEXT NOT NULL,
                    last_update_time_seconds REAL NOT NULL
                )
            ''')
            await db.commit()

    async def get_timer_from_db(self, thread_id: str) -> dict:
        async with aiosqlite.connect(self.timer_db_path) as db:
            async with db.execute(
                'SELECT timer_data, last_update_time_seconds FROM update_timers WHERE thread_id = ?', 
                (thread_id,)
            ) as cursor:
                result = await cursor.fetchone()
                if result:
                    return {
                        "thread_id": thread_id,
                        "update_timers": json.loads(result[0]),
                        "last_update_time_seconds": result[1]
                    }
                else:
                    # 默认值
                    return {
                        "thread_id": thread_id,
                        "update_timers": [
                            {'left': 0.0, 'threshold': 5.0, 'stable_time_range': [{'$gte': 0.0}, {'$lt': 43200.0}]},
                            {'left': 0.0, 'threshold': 30.0, 'stable_time_range': [{'$gte': 43200.0}, {'$lt': 86400.0}]},
                            {'left': 0.0, 'threshold': 60.0, 'stable_time_range': [{'$gte': 86400.0}, {'$lt': 864000.0}]},
                            {'left': 0.0, 'threshold': 500.0, 'stable_time_range': [{'$gte': 864000.0}, {'$lt': 8640000.0}]},
                            {'left': 0.0, 'threshold': 3600.0, 'stable_time_range': [{'$gte': 8640000.0}]},
                        ],
                        "last_update_time_seconds": now_seconds()
                    }

    async def set_timer_to_db(self, thread_id: str, update_timers: list, last_update_time_seconds: float) -> None:
        async with aiosqlite.connect(self.timer_db_path) as db:
            await db.execute('''
                INSERT OR REPLACE INTO update_timers (thread_id, timer_data, last_update_time_seconds)
                VALUES (?, ?, ?)
            ''', (thread_id, json.dumps(update_timers), last_update_time_seconds))
            await db.commit()


    async def update_timer(self, thread_id: str):
        types = ['original', 'summary', 'semantic']

        update_count = 0
        data = await self.get_timer_from_db(thread_id)

        current_time_seconds = now_seconds()
        time_diff = current_time_seconds - data['last_update_time_seconds']
        data['last_update_time_seconds'] = current_time_seconds
        for timer in data['update_timers']:
            timer['left'] += time_diff
            if timer['left'] >= timer['threshold']:
                timer['left'] = 0
                for t in types:
                    where = validated_where({'$and': [{'stable_time': item} for item in timer['stable_time_range']]})
                    result = await self.aget(
                        thread_id=thread_id,
                        memory_type=t,
                        where=where,
                        include=['metadatas']
                    )
                    if result['ids']:
                        await self.update_memories(result, t, thread_id)
                        update_count += len(result['ids'])

        await self.set_timer_to_db(data['thread_id'], data['update_timers'], data['last_update_time_seconds'])
        print(f'updated {update_count} memories for thread "{data['thread_id']}".')


    def update(self, metadata: dict, current_time_seconds: float) -> dict:
        """
        更新记忆的可检索性（模拟时间流逝）
        """
        #retrievability = metadata["retrievability"] * math.exp(-delta_t / metadata["stability"])

        if metadata["stable_time"] == 0.0:
            print('意外的stable_time为0，metadata：' + str(metadata))
            return {"forgot": True}

        x = (current_time_seconds - metadata["last_accessed_time_seconds"]) / metadata["stable_time"]
        if x >= 1:
            return {"forgot": True}
        retrievability = 1 - x ** 0.4

        return {"retrievability": retrievability}

    def recall(self, metadata: dict, current_time_seconds: float, time_settings: AgentTimeSettings, strength: float = 1.0) -> dict:
        """
        调用记忆时重置可检索性并增强稳定性
        """
        # 更新可检索性
        updated_metadata = self.update(metadata, current_time_seconds)
        if updated_metadata.get("forgot"):
            return {"forgot": True}

        datetime_alpha = calculate_memory_datetime_alpha(seconds_to_datetime(current_time_seconds).astimezone(get_agent_time_zone(time_settings)))
        stable_strength = calculate_stability_curve(updated_metadata["retrievability"])
        stable_time_diff = metadata["stable_time"] * stable_strength - metadata["stable_time"]
        if stable_time_diff >= 0:
            stable_time_diff = stable_time_diff * metadata["difficulty"] * strength * datetime_alpha
        stable_time = metadata["stable_time"] + stable_time_diff

        retrievability = min(1.0, updated_metadata["retrievability"] + strength * datetime_alpha)

        metadata_patch = {
            "last_accessed_time_seconds": current_time_seconds,
            "retrievability": retrievability,
            "stable_time": stable_time
        }

        if metadata["difficulty"] < 1.0:
            difficulty = min(1.0, metadata["difficulty"] + stable_strength * metadata["difficulty"] * strength * datetime_alpha * 0.5)
            metadata_patch["difficulty"] = difficulty

        return metadata_patch

    async def update_memories(self, results: dict[str, Any], memory_type: str, thread_id: str) -> None:
        current_time_seconds = now_seconds()
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
            metadata_patch = self.update(metadatas[i], current_time_seconds)
            if metadata_patch.get("forgot"):
                ids_to_delete.add(ids[i])
            else:
                ids_new.append(ids[i])
                metadatas_new.append(metadata_patch)
        if ids_to_delete:
            self.get_collection(thread_id, memory_type).delete(ids=list(ids_to_delete))
        await self.aupdate_metadatas(ids_new, metadatas_new, memory_type, thread_id)


    class InitialMemory(BaseModel):
        content: str = Field(description="The content of the memory")
        creation_time_seconds: float = Field(description="The creation time seconds of the memory")
        type: Literal["original", "summary", "semantic"] = Field(description="The type of the memory")
        id: str = Field(default_factory=lambda: str(uuid4()), description="The id of the memory")
        stable_time: float = Field(description="The stable time base of the memory", gt=0.0)

    async def add_memories(self, memories: list[InitialMemory], thread_id: str, time_settings: AgentTimeSettings) -> None:
        docs: dict[str, list[Document]] = {
            "original": [],
            "summary": [],
            "semantic": []
        }
        current_time_seconds = now_seconds()
        for memory in memories:
            # 使用jieba对memory.content进行分词并过滤掉重复词
            words = set(jieba.cut(memory.content))
            max_words_length = 200
            difficulty = 1 - min(1.0, (len(words) / max_words_length) ** 3)
            datetime_alpha = calculate_memory_datetime_alpha(seconds_to_datetime(memory.creation_time_seconds).astimezone(get_agent_time_zone(time_settings)))
            stable_time = memory.stable_time * difficulty * datetime_alpha # 稳定性，决定了可检索性的衰减速度
            metadata = {
                "creation_time_seconds": memory.creation_time_seconds,
                "stable_time": stable_time,
                "retrievability": 1.0, # 可检索性，决定了检索的概率
                "difficulty": difficulty, # 难度，决定了稳定性基数增长的多少。可能会出现无法长期保留的记忆，如整本书的内容。
                "last_accessed_time_seconds": memory.creation_time_seconds,
                "type": memory.type
            }
            r = self.update(metadata, current_time_seconds)
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
                await self.aadd_documents(docs[t], thread_id, t)



    class RetrieveInput(BaseModel):
        search_string: Optional[str] = Field(default=None, description="检索字符串")
        search_type: Literal["original", "summary", "semantic"] = Field(description="检索类型")
        search_method: Literal["similarity", "mmr"] = Field(default="similarity", description="检索方法")
        k: int = Field(description="返回的记忆数量", ge=0)
        fetch_k: int = Field(description="从多少个结果中筛选出最终结果(fetch_k>k)，目前只有mmr使用", ge=0)
        similarity_weight: float = Field(description="相似性权重", ge=0, le=1),
        retrievability_weight: float = Field(description="可检索性权重", ge=0, le=1),
        diversity_weight: float = Field(description="多样性权重，这是mmr的参数", ge=0, le=1)
        metadata_filter: list = Field(default_factory=list, description="元数据过滤条件")
        strength: float = Field(description="检索强度，作为stable_time和retrievability的乘数", ge=0)
    async def retrieve_memories(self,
                                inputs: list[RetrieveInput],
                                thread_id: str,
                                agent_time_settings: AgentTimeSettings
                                ) -> list[tuple[Document, float]]:
        """检索记忆向量库中的文档并返回排序结果。"""

        # 构建有效输入字典：过滤掉无效输入（无搜索字符串或k<=0）
        #inputs: list[dict] = [i.model_dump() for i in inputs]
        inputs_dict: list[dict] = []
        gets = []
        for i in inputs:
            if i.k <= 0 or i.fetch_k < i.k:
                continue
            elif not i.search_string:
                if i.metadata_filter:
                    gets.append(i.model_dump())
                else:
                    continue
            else:
                inputs_dict.append(i.model_dump())
        inputs: list[dict] = inputs_dict


        # 至少需要一个有效输入参数
        if not inputs and not gets:
                raise ValueError("没有要检索的记忆！")

        combined_docs_and_scores: list[tuple[Document, float]] = []

        current_time_seconds = now_seconds()

        ids_to_delete = {"original": set(), "summary": set(), "semantic": set()}

        # 这里是纯时间过滤的检索
        if gets:
            for g in gets:
                results = await self.aget(
                    thread_id=thread_id,
                    memory_type=g["search_type"],
                    where=validated_where({"$and": g["metadata_filter"]}),
                    limit=g["k"],
                )
                docs = get_results_to_docs(results)
                ids = []
                metadatas = []
                strength = g["strength"]
                for doc in docs:
                    patched_metadata = self.recall(
                        metadata=doc.metadata,
                        current_time_seconds=current_time_seconds,
                        time_settings=agent_time_settings,
                        strength=strength
                    )
                    if patched_metadata.get("forgot"):
                        ids_to_delete[g["search_type"]].add(doc.id)
                    else:
                        ids.append(doc.id)
                        metadatas.append(patched_metadata)
                        doc.metadata.update(patched_metadata)
                await self.aupdate_metadatas(ids, metadatas, g["search_type"], thread_id)
                combined_docs_and_scores.extend([(doc, doc.metadata["retrievability"]) for doc in docs if doc.id not in ids_to_delete[g["search_type"]]])
    
        if not inputs:
            return combined_docs_and_scores


        search_strings = []
        last_input = {}

        # 收集唯一搜索字符串并标记重复项
        for input in inputs:
            if last_input:

                # 如果当前搜索字符串与前一个相同，则标记为重复项
                if input["search_string"] == last_input["search_string"]:
                    input["same_as_last"] = True
                else:
                    search_strings.append(input["search_string"])
            else:
                search_strings.append(input["search_string"])
            last_input = input

        # 批量生成嵌入向量：多个字符串使用aembed_documents，单个使用aembed_query
        if len(search_strings) > 1:
            search_embeddings = await self.embeddings.aembed_documents(search_strings)
        else:
            search_embeddings = [await self.embeddings.aembed_query(search_strings[0])]

        embeddings_index = 0
        first_loop = True

        # 将嵌入向量分配给对应的输入类型
        for input in inputs:
            if first_loop:
                first_loop = False
            elif not input.get("same_as_last"):
                embeddings_index += 1
            input["search_embedding"] = search_embeddings[embeddings_index]


        current_time_seconds = now_seconds()

        # 对每个输入类型执行向量搜索
        for input in inputs:
            if input["search_method"] == "similarity":
                docs_and_scores = await self.asimilarity_search_by_vector_with_score_and_retrievability(
                    embedding=input["search_embedding"],
                    thread_id=thread_id,
                    memory_type=input["search_type"],
                    k=input["k"],
                    similarity_weight=input["similarity_weight"],
                    retrievability_weight=["retrievability_weight"],
                    filter=validated_where({"$and": input.get("metadata_filter", [])}) if input.get("metadata_filter") else None
                )
            elif input["search_method"] == "mmr":
                docs_and_scores = await self.amax_marginal_relevance_search_by_vector_with_retrievability(
                    embedding=input["search_embedding"],
                    thread_id=thread_id,
                    memory_type=input["search_type"],
                    k=input["k"],
                    fetch_k=input["fetch_k"],
                    similarity_weight=input["similarity_weight"],
                    retrievability_weight=input["retrievability_weight"],
                    diversity_weight=input["diversity_weight"],
                    filter=validated_where({"$and": input.get("metadata_filter", [])}) if input.get("metadata_filter") else None
                )
            ids = []
            metadatas = []
            strength = input["strength"]
            for doc, score in docs_and_scores:
                patched_metadata = self.recall(
                    metadata=doc.metadata,
                    current_time_seconds=current_time_seconds,
                    time_settings=agent_time_settings,
                    strength=strength
                )
                if patched_metadata.get("forgot"):
                    ids_to_delete[input["search_type"]].add(doc.id)
                else:
                    ids.append(doc.id)
                    metadatas.append(patched_metadata)
            await self.aupdate_metadatas(ids, metadatas, input["search_type"], thread_id)
            combined_docs_and_scores.extend([_ for _ in docs_and_scores if _[0].id not in ids_to_delete[input["search_type"]]])


        for memory_type, delete_ids in ids_to_delete.items():
            if delete_ids:
                self.get_collection(thread_id, memory_type).delete(ids=list(delete_ids))


        # 按照score降序排序（score越大索引越小）
        combined_docs_and_scores.sort(key=lambda x: x[1], reverse=True)

        #TODO：可以做一个最大token限制，如果超出限制，则删除掉token最多的文档再试一次


        #combined_docs = [doc for doc, score in combined_docs_and_scores] + combined_docs


        return combined_docs_and_scores


    def update_metadatas(self, ids: list[str], metadatas: list[dict], memory_type: str, thread_id: str) -> None:
        if not ids:
            return
        if len(ids) != len(metadatas):
            raise ValueError("ids and metadatas must have the same length")
        self.get_collection(thread_id, memory_type).update(ids=ids, metadatas=metadatas)

    async def aupdate_metadatas(self, ids: list[str], metadatas: list[dict], memory_type: str, thread_id: str) -> None:
        await run_in_executor(None, self.update_metadatas, ids, metadatas, memory_type, thread_id)

    def get(
        self,
        thread_id: str,
        memory_type: str,
        ids: Optional[OneOrMany[ID]] = None,
        where: Optional[Where] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        where_document: Optional[WhereDocument] = None,
        include: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        kwargs = {
            "ids": ids,
            "where": where if where else None,
            "limit": limit,
            "offset": offset,
            "where_document": where_document if where_document else None,
        }

        if include is not None:
            kwargs["include"] = include
        return self.get_collection(thread_id, memory_type).get(**kwargs)


    async def aget(
        self,
        thread_id: str,
        memory_type: str,
        ids: Optional[OneOrMany[ID]] = None,
        where: Optional[Where] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        where_document: Optional[WhereDocument] = None,
        include: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        return await run_in_executor(None, self.get, thread_id, memory_type, ids, where, limit, offset, where_document, include)

    async def aadd_documents(
        self,
        documents: list[Document],
        thread_id: str,
        memory_type: str
    ) -> None:
        contents = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.id if doc.id else str(uuid4()) for doc in documents]
        embeddings = await self.embeddings.aembed_documents(contents)
        collection = self.get_collection(thread_id, memory_type)
        await run_in_executor(None, collection.add, ids, embeddings, metadatas, contents, None, None)


    def similarity_search_by_vector_with_score(
        self,
        thread_id: str,
        memory_type: str,
        embedding: list[float],
        k: int = 5,
        filter: Optional[dict[str, str]] = None,
        where_document: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        collection = self.get_collection(thread_id, memory_type)
        result = collection.query(
            query_embeddings=embedding,
            n_results=k,
            where=filter if filter else None,
            where_document=where_document if where_document else None,
            **kwargs,
        )
        return results_to_docs_and_scores(result)
    
    async def asimilarity_search_by_vector_with_score(
        self,
        thread_id: str,
        memory_type: str,
        embedding: list[float],
        k: int = 5,
        filter: Optional[dict[str, str]] = None,
        where_document: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return await run_in_executor(None, self.similarity_search_by_vector_with_score, thread_id, memory_type, embedding, k, filter, where_document, **kwargs)



    async def asimilarity_search_by_vector_with_score_and_retrievability(
        self,
        embedding: list[float],
        thread_id: str,
        memory_type: str,
        k: int = 5,
        similarity_weight: float = 0.6,
        retrievability_weight: float = 0.4,
        filter: Optional[dict[str, str]] = None,
        where_document: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        #对于chroma的cosine来说，输出的score范围是0~1，越小越相似。这里统一反转为越大越相似
        docs_and_scores = await self.asimilarity_search_by_vector_with_score(
            thread_id=thread_id,
            memory_type=memory_type,
            embedding=embedding,
            k=k,
            filter=filter,
            where_document=where_document,
            **kwargs)
        if not docs_and_scores:
            return []
        docs_and_scores_with_retrievability = [(doc, (1 - score) * similarity_weight + doc.metadata.get("retrievability", 0) * retrievability_weight) for doc, score in docs_and_scores]
        docs_and_scores_with_retrievability.sort(key=lambda x: x[1], reverse=True)
        return docs_and_scores_with_retrievability


    async def amax_marginal_relevance_search_with_retrievability(
        self,
        query: str,
        thread_id: str,
        memory_type: str,
        k: int = 5,
        fetch_k: int = 20,
        similarity_weight: float = 0.4,
        retrievability_weight: float = 0.3,
        diversity_weight: float = 0.3,
        filter: Optional[dict[str, str]] = None,
        where_document: Optional[dict[str, str]] = None,
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
            thread_id=thread_id,
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
        thread_id: str,
        memory_type: str,
        embedding: list[float],
        k: int = 5,
        fetch_k: int = 20,
        similarity_weight: float = 0.4,
        retrievability_weight: float = 0.3,
        diversity_weight: float = 0.3,
        filter: Optional[dict[str, str]] = None,
        where_document: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_by_vector_with_retrievability,
            thread_id,
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
        thread_id: str,
        memory_type: str,
        embedding: list[float],
        k: int = 5,
        fetch_k: int = 20,
        similarity_weight: float = 0.4,
        retrievability_weight: float = 0.3,
        diversity_weight: float = 0.3,
        filter: Optional[dict[str, str]] = None,
        where_document: Optional[dict[str, str]] = None,
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
        collection = self.get_collection(thread_id, memory_type)
        if collection is None:
            raise ValueError(f"Collection {thread_id}_{memory_type} does not exist or cannot create.")
        results = collection.query(
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

        candidates = results_to_docs(results)

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
        thread_id: str,
        memory_type: Literal["original", "semantic", "summary"],
    ) -> Collection:
        return self.vector_store_client.get_or_create_collection(
            name=f"{thread_id}_{memory_type}",
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )

def validated_where(where: dict) -> Optional[dict]:
    '''只解决在and或or时列表里不能只有一个元素的问题'''
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


def results_to_docs(results: Any) -> list[Document]:
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

def get_results_to_docs(results: dict[str, Any]) -> list[Document]:
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

def results_to_docs_and_scores(results: Any) -> list[tuple[Document, float]]:
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