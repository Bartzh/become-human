from typing import Literal
from pydantic import Field, BaseModel


PLUGIN_NAME = "bh_memory"

AnyMemoryType = Literal["original", "reflective", "summary"]
MEMORY_TYPES: list[AnyMemoryType] = ["original", "reflective", "summary"]

class MemoryRetrievalConfig(BaseModel):
    class Retrieval(BaseModel):
        memory_type: AnyMemoryType = Field(description="要检索的记忆类型")
        k: int = Field(gt=0, description="检索返回的记忆数量")
        fetch_k: int = Field(default=100, gt=0, description="从多少个结果中筛选出最终的结果")
        depth: int = Field(default=0, ge=0, description="检索的记忆深度，会在0之间随机取值，指被检索记忆的相邻n个记忆也会被召回")
        retrieve_method: Literal['similarity', 'mmr'] = Field(default='mmr', description="检索排序算法：[similarity, mmr]")
        similarity_weight: float = Field(default=0.35, description="检索权重：相似性权重，范围[0,1]，总和需为1", ge=0.0, le=1.0)
        retrievability_weight: float = Field(default=0.35, description="检索权重：可访问性权重，范围[0,1]，总和需为1", ge=0.0, le=1.0)
        diversity_weight: float = Field(default=0.3, description="检索权重：多样性权重，范围[0,1]。只在检索方法为mmr时生效，总和需为1", ge=0.0, le=1.0)

    # k: int = Field(default=16, ge=0, description="检索返回的记忆数量")
    # fetch_k: int = Field(default=250, ge=0, description="从多少个结果中筛选出最终的结果")
    # depth: int = Field(default=2, ge=0, description="检索的记忆深度，会在0之间随机取值，指被检索记忆的相邻n个记忆也会被召回")
    # original_ratio: float = Field(default=2.0, description="检索结果中原始记忆出现的初始比例", ge=0.0)
    # reflective_ratio: float = Field(default=5.0, description="检索结果中反思记忆出现的初始比例", ge=0.0)
    # summary_ratio: float = Field(default=1.0, description="检索结果中总结记忆出现的初始比例", ge=0.0)
    # search_method: Literal['similarity', 'mmr'] = Field(default='mmr', description="检索排序算法：[similarity, mmr]")
    # similarity_weight: float = Field(default=0.5, description="检索权重：相似性权重，范围[0,1]，总和需为1", ge=0.0, le=1.0)
    # retrievability_weight: float = Field(default=0.25, description="检索权重：可访问性权重，范围[0,1]，总和需为1", ge=0.0, le=1.0)
    # diversity_weight: float = Field(default=0.25, description="检索权重：多样性权重，范围[0,1]。只在检索方法为mmr时生效，总和需为1", ge=0.0, le=1.0)

    retrievals: list[Retrieval] = Field(default_factory=list, description="要检索的记忆类型")
    stable_k: int = Field(default=5, ge=0, description="最终显示几个完整的记忆，剩下的记忆会被简略化")
    strength: float = Field(default=1.0, description="检索强度，作为倍数将乘以被检索记忆的可检索性、稳定时长与难易度的提升幅度，范围[0,1]，作为主动检索时固定为1")
    weight: float = Field(default=1.0, ge=0, description="若有多个config，随机选中此config的概率权重（random.choices(configs, weights)）")
