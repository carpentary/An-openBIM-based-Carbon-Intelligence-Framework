from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import CQMatch, EntityPointer


MSCA = "http://bimlca.uni.edu/ontology#"

DEFAULT_ALIASES: dict[str, tuple[str, str]] = {
    "metro station": (MSCA + "MetroStationProject", "Metro station project"),
    "地铁车站": (MSCA + "MetroStationProject", "地铁车站项目"),
    "project": (MSCA + "ConstructionProject", "Construction project"),
    "项目": (MSCA + "ConstructionProject", "项目"),
    "diaphragm wall": (MSCA + "Type_DiaphragmWall", "Diaphragm wall"),
    "地下连续墙": (MSCA + "Type_DiaphragmWall", "地下连续墙"),
    "地连墙": (MSCA + "Type_DiaphragmWall", "地下连续墙"),
    "top slab": (MSCA + "Type_TopSlab", "Top slab"),
    "顶板": (MSCA + "Type_TopSlab", "顶板"),
    "middle slab": (MSCA + "Type_MiddleSlab", "Middle slab"),
    "中板": (MSCA + "Type_MiddleSlab", "中板"),
    "bottom slab": (MSCA + "Type_BottomSlab", "Bottom slab"),
    "底板": (MSCA + "Type_BottomSlab", "底板"),
    "side wall": (MSCA + "Type_SideWall", "Side wall"),
    "侧墙": (MSCA + "Type_SideWall", "侧墙"),
    "beam": (MSCA + "Beam", "Beam"),
    "梁": (MSCA + "Beam", "梁"),
    "column": (MSCA + "Column", "Column"),
    "柱": (MSCA + "Column", "柱"),
    "pile": (MSCA + "Pile", "Pile"),
    "桩": (MSCA + "Pile", "桩"),
    "concrete": (MSCA + "Concrete", "Concrete"),
    "混凝土": (MSCA + "Concrete", "混凝土"),
    "steel": (MSCA + "Steel", "Steel"),
    "钢": (MSCA + "Steel", "钢"),
    "ggbs": (MSCA + "GroundGranulatedBlastFurnaceSlag", "GGBS"),
    "slag": (MSCA + "GroundGranulatedBlastFurnaceSlag", "GGBS"),
    "矿渣": (MSCA + "GroundGranulatedBlastFurnaceSlag", "矿渣粉"),
    "work package": (MSCA + "ProjectWorkPackage", "Project work package"),
    "工序": (MSCA + "ProjectWorkPackage", "项目工作包"),
    "resource": (MSCA + "ResourceInstance", "Resource instance"),
    "资源": (MSCA + "ResourceInstance", "资源实例"),
    "emission factor": (MSCA + "EFD_Factor", "Emission factor"),
    "排放因子": (MSCA + "EFD_Factor", "排放因子"),
    "cpd": (MSCA + "CPD_Database", "Construction process database"),
    "efd": (MSCA + "EFD_Database", "Emission factor database"),
    "a1-a3": (MSCA + "A1-A3_ProductStage", "A1-A3 product stage"),
    "a4": (MSCA + "A4_TransportationStage", "A4 transportation stage"),
    "a5": (MSCA + "A5_ConstructionStage", "A5 construction stage"),
}


@dataclass(frozen=True)
class CQEntry:
    cq_id: str
    group: str
    question_en: str
    question_zh: str
    validation_objective: str
    semantic_path: str
    implementation_layers: str
    support_status: str
    dependency_or_boundary: str
    expected_outcome: str
    sparql_file: str
    fixture_min_rows: int

    @property
    def searchable_text(self) -> str:
        return " ".join(
            (
                self.cq_id,
                self.group,
                self.question_en,
                self.question_zh,
                self.validation_objective,
                self.semantic_path,
                self.expected_outcome,
            )
        )


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", lowered)
    for run in re.findall(r"[\u3400-\u9fff]+", lowered):
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return tokens


def _character_ngrams(text: str, size: int = 3) -> Counter[str]:
    compact = "".join(re.findall(r"[a-z0-9\u3400-\u9fff]+", text.lower()))
    if len(compact) <= size:
        return Counter({compact: 1}) if compact else Counter()
    return Counter(compact[index : index + size] for index in range(len(compact) - size + 1))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def validate_semantic_path(path: str) -> bool:
    """Reject direct Element→Resource/Carbon shortcuts forbidden by S-E-P-R-C."""
    compact = re.sub(r"\s+", " ", path).lower()
    forbidden = (
        r"\belement\b\s*--[a-z0-9_:/|*\-]+-->\s*\bresource(?:instance)?\b",
        r"\belement\b\s*--[a-z0-9_:/|*\-]+-->\s*\bcarbon(?:emission)?\b",
        r"\bphysicalelement\b\s*--[a-z0-9_:/|*\-]+-->\s*\bresource(?:instance)?\b",
    )
    return not any(re.search(pattern, compact) for pattern in forbidden)


def governed_cq_candidates(query: str) -> list[str]:
    """Map explicit project-query intents to the smallest governed CQ set.

    The rules are CQ-semantic expansions, not answer lookup: they only select
    catalogue questions and never read project values or ground-truth answers.
    Unrecognised queries fall back to the lexical/character retriever.
    """

    text = query.lower().replace("–", "-").replace("—", "-")

    def has(*terms: str) -> bool:
        return any(term in text for term in terms)

    ontology_query = has("ontology term", "ontology术语", "ontology 术语", "本体术语")
    transport = has("a4", "transport", "haul", "t-km", "tkm", "运输", "运距", "吨公里")
    foundation = has("foundation pit", "foundation-pit", "earthwork", "excavation", "backfill", "基坑", "土方", "开挖", "回填")
    enclosure = has(
        "enclosure", "diaphragm", "sectionsteelstrut", "steel_beam", "steel_support",
        "applicable-element mapping", "applicable elements are unmapped",
        "围护", "地下连续墙", "地连墙", "适用围护构件", "映射率", "未映射",
    )
    main_structure = has("main structure", "main-structure", "主体结构")
    missing = has("without", "omit", "missing", "不提供", "缺少", "省略")
    persistent = has(
        "persist",
        "commit",
        "write to",
        "write a new",
        "write an alternative",
        "permanently",
        "correct the graph",
        "update the graph",
        "永久",
        "持久化",
        "提交",
        "写入",
    )
    sensitivity = has("sensitivity analysis", "parameter sweep", "敏感性分析", "参数扫描")
    reuse = has("reuse cycle", "reuse cycles", "formwork reuse", "周转次数", "模板周转")
    ggbs = has("ggbs", "slag", "矿渣", "矿粉")
    factor_change = has(
        "emission-factor reduction",
        "emission factor is reduced",
        "factor is reduced",
        "reduction in the concrete emission factor",
        "reduction in the steel emission factor",
        "排放因子降低",
    )
    intensity = has("carbon intensity", "kgco2e/m2", "kgco2e/m3", "碳强度")
    scenario = has(
        "simulate", "scenario", "what if", "reduced by", "substitution", "calculate a4",
        "sensitivity analysis", "reuse cycle", "formwork reuse",
        "模拟", "情景", "若", "如果", "替代", "降低", "计算50 t", "计算50t", "敏感性分析", "周转次数",
    )
    trace = has("trace", "provenance", "audit chain", "追溯", "溯源", "完整链路")
    cpd_code = has("cpd code", "cpd work-package", "cpd工作包", "cpd编码", "cpd 编码")
    out_of_scope = bool(re.search(r"\b(?:b1|c1|module\s*d)\b", text)) or has(
        "operational-use", "deconstruction", "beyond the system boundary", "运营使用", "拆除阶段", "边界外", "d模块"
    )

    if re.search(r"\bcq-[a-z]\d{2}\b", text):
        return [re.search(r"\bcq-[a-z]\d{2}\b", text).group(0).upper()]
    if out_of_scope:
        return ["CQ-S02", "CQ-S05", "CQ-Q04"]
    if persistent:
        if transport:
            return ["CQ-X02", "CQ-X07", "CQ-X08"] if missing else ["CQ-X02", "CQ-X07"]
        if ggbs:
            return ["CQ-X01", "CQ-X07"]
        if reuse:
            return ["CQ-X03", "CQ-X07"]
        if sensitivity:
            return ["CQ-X04", "CQ-X07"]
        if factor_change or has("emission factor", "排放因子"):
            return ["CQ-X05", "CQ-X07"]
        return ["CQ-X07"]
    if scenario and missing:
        if transport:
            return ["CQ-X02", "CQ-X08", "CQ-Q04"]
        if ggbs:
            return ["CQ-X01", "CQ-X08", "CQ-Q04"]
        if reuse:
            return ["CQ-X03", "CQ-X08", "CQ-Q04"]
        if sensitivity:
            return ["CQ-X04", "CQ-X08", "CQ-Q04"]
        if factor_change or has("emission factor", "排放因子"):
            return ["CQ-X05", "CQ-X08", "CQ-Q04"]
    if scenario and sensitivity:
        if foundation:
            return ["CQ-X04", "CQ-C05"]
        if transport:
            return ["CQ-X04", "CQ-C06"]
        return ["CQ-X04", "CQ-C04"]
    if scenario and transport:
        return ["CQ-X02", "CQ-C06"]
    if ggbs and scenario:
        return ["CQ-X01"]
    if scenario and reuse:
        return ["CQ-X03"]
    if scenario and (factor_change or has("emission factor", "因子降低", "排放因子")):
        return ["CQ-X05"]

    if intensity:
        if foundation:
            return ["CQ-C05", "CQ-C10"]
        if has("project", "项目"):
            return ["CQ-C04", "CQ-C10"]
        return ["CQ-C10"]
    if has("calculate material emission", "calculate main-structure equipment emission"):
        return ["CQ-C02"]
    if has(
        "public geometry",
        "station form",
        "construction zones",
        "ground conditions",
        "production peaks",
    ):
        return ["CQ-S01"]
    if transport and has("mapping coverage", "mapped, pending and excluded", "映射覆盖率"):
        return ["CQ-C06", "CQ-Q04"]
    if transport and has("compare", "comparison") and sum((main_structure, enclosure, foundation)) >= 2:
        return ["CQ-C03", "CQ-C06"]

    if ontology_query:
        if trace or has("system-element-process-resource-carbon", "system-element", "system–element"):
            return ["CQ-C02", "CQ-P03", "CQ-P04", "CQ-P06"]
        if transport:
            return ["CQ-C06", "CQ-X02"]
        if has("main, enclosure", "main、enclosure", "main, enclosure and foundation", "主体、围护", "主体、围护和基坑", "by main"):
            return ["CQ-C03", "CQ-C08"]
        return ["CQ-C03", "CQ-C04"]

    if trace:
        return ["CQ-C02", "CQ-P03", "CQ-P04", "CQ-P06"]
    if main_structure and has("concrete resource", "concrete resources", "混凝土资源") and has("breakdown", "明细"):
        return ["CQ-P04", "CQ-C03"]
    if main_structure and has("work-package template", "work package template", "工作包模板") and has("ten", "10", "十个", "最多"):
        return ["CQ-I02", "CQ-P03"]
    if main_structure and has("element", "physical element", "构件") and has("highest", "largest", "ten", "10", "最高", "最大", "十个"):
        return ["CQ-C09"]

    if has("formula", "公式"):
        return ["CQ-C03", "CQ-C04", "CQ-C05", "CQ-C06"]
    if has("each a1-a5 stage", "a1-a5 stage", "各阶段占", "各a1-a5阶段"):
        return ["CQ-C03", "CQ-C04"]
    if has("lifecycle stages are included", "included in the active assessment", "核算包含", "包含哪些生命周期"):
        return ["CQ-S02", "CQ-S05"]
    if has("applicability scope", "适用范围"):
        return ["CQ-S02"]
    if has("ifc", "ifc product", "ifc products", "ifc产品", "ifc 产品") and has("product", "products", "产品") and has("main", "enclosure", "主体", "围护"):
        return ["CQ-S01", "CQ-S03"]

    foundation_process = foundation and has("work package", "work packages", "resource flow", "resource flows", "工作包", "资源流")
    foundation_geometry = foundation and has(
        "volume", "envelope", "parameter", "sensitivity result", "体积", "总体积", "工程量模型", "敏感性结果"
    )
    if foundation and cpd_code:
        return ["CQ-P01", "CQ-P02", "CQ-C05"]
    if foundation and has("haul distance", "off-site haul", "外运采用的运距", "土方外运"):
        return ["CQ-C05", "CQ-C06"]
    if foundation_process or foundation_geometry:
        return ["CQ-C05"]

    carbon_or_stage = has("carbon", "emission", "a1-a3", "a1-a5", "a5", "碳", "排放", "阶段")
    project_stage = has("project", "项目") and carbon_or_stage and (
        has("a1-a3", "a1-a5", "a5", "project a4", "project carbon", "项目a4", "项目碳", "各阶段")
        or has("lifecycle stage", "生命周期阶段")
    )
    multiple_scopes = (
        sum((main_structure, enclosure, foundation)) >= 2
        or has("three engineering scopes", "each engineering scope", "三个工程范围", "各工程范围")
    )
    if project_stage and not has("engineering scope", "工程范围"):
        return ["CQ-C03", "CQ-C04"]
    system_stage_or_total = has(
        "total carbon", "total carbon emission", "carbon emission of", "a1-a3", "a1-a5", "a5 emissions",
        "碳排放合计", "碳排放最高", "总碳排放", "a1-a3、a4和a5", "a1-a3、a4 和 a5",
    )
    if (multiple_scopes and not transport) or (system_stage_or_total and (main_structure or enclosure or foundation)):
        return ["CQ-C03", "CQ-C08"]

    if transport:
        material_code = bool(re.search(r"\bm-\d+-\d+\b", text))
        if has("assigned by default", "mass conversion", "默认配置", "质量换算") or (
            material_code and has("default", "默认")
        ):
            return ["CQ-P05", "CQ-C06"]
        return ["CQ-C06"]

    if carbon_or_stage and (multiple_scopes or main_structure or enclosure or (foundation and has("total", "stage", "合计", "阶段", "排放"))):
        return ["CQ-C03", "CQ-C08"]

    if foundation:
        if cpd_code:
            return ["CQ-P01", "CQ-P02", "CQ-C05"]
        return ["CQ-C05"]

    if enclosure:
        if cpd_code and has("diaphragm", "地下连续墙", "地连墙"):
            return ["CQ-I03", "CQ-P01", "CQ-P02"]
        if has("diaphragm", "地下连续墙", "地连墙") and has("volume", "length", "体积", "长度"):
            return ["CQ-I03", "CQ-P03", "CQ-C08"]
        return ["CQ-I02", "CQ-I03", "CQ-P01"]

    if has("project", "项目") and carbon_or_stage:
        return ["CQ-C03", "CQ-C04"]
    return []


class CQCatalogue:
    """Hybrid BM25/character retrieval over the governed 44-CQ catalogue."""

    def __init__(self, entries: Iterable[CQEntry]):
        self.entries = tuple(entries)
        if not self.entries:
            raise ValueError("CQ catalogue is empty")
        self._documents = [_tokens(entry.searchable_text) for entry in self.entries]
        self._term_frequencies = [Counter(document) for document in self._documents]
        self._doc_frequency: Counter[str] = Counter()
        for document in self._documents:
            self._doc_frequency.update(set(document))
        self._average_length = sum(map(len, self._documents)) / len(self._documents)
        self._ngrams = [_character_ngrams(entry.searchable_text) for entry in self.entries]

    @classmethod
    def load(cls, path: Path) -> "CQCatalogue":
        entries: list[CQEntry] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                entries.append(
                    CQEntry(
                        cq_id=row["CQ_ID"],
                        group=row["Group"],
                        question_en=row["Question_EN"],
                        question_zh=row["Question_ZH"],
                        validation_objective=row["Validation_Objective"],
                        semantic_path=row["Core_Semantic_Path"],
                        implementation_layers=row["Implementation_Layers"],
                        support_status=row["Support_Status"],
                        dependency_or_boundary=row["Dependency_or_Boundary"],
                        expected_outcome=row["Expected_Outcome"],
                        sparql_file=row["SPARQL_File"],
                        fixture_min_rows=int(row["Fixture_Min_Rows"]),
                    )
                )
        return cls(entries)

    def group_counts(self) -> dict[str, int]:
        return dict(Counter(entry.group for entry in self.entries))

    def by_id(self, cq_id: str) -> CQEntry:
        normalised = cq_id.upper()
        for entry in self.entries:
            if entry.cq_id.upper() == normalised:
                return entry
        raise KeyError(cq_id)

    def search(self, query: str, limit: int = 5, *, governed_intents: bool = True) -> list[CQMatch]:
        query_tokens = _tokens(query)
        query_ngrams = _character_ngrams(query)
        bm25_scores = [self._bm25(query_tokens, index) for index in range(len(self.entries))]
        bm25_peak = max(bm25_scores, default=0.0)
        exact_id = re.search(r"\bCQ-[A-Z]\d{2}\b", query.upper())
        governed_ids = governed_cq_candidates(query) if governed_intents else []
        governed_rank = {cq_id: index for index, cq_id in enumerate(governed_ids)}
        scored: list[tuple[float, CQEntry]] = []
        for index, entry in enumerate(self.entries):
            lexical = bm25_scores[index] / bm25_peak if bm25_peak else 0.0
            dense_proxy = _cosine(query_ngrams, self._ngrams[index])
            score = 0.62 * lexical + 0.38 * dense_proxy
            if exact_id and exact_id.group(0) == entry.cq_id:
                score = 1.0
            elif governed_ids and entry.cq_id in governed_rank:
                semantic_prior = 1.0 - 0.03 * governed_rank[entry.cq_id]
                score = 0.82 * semantic_prior + 0.12 * lexical + 0.06 * dense_proxy
            scored.append((min(1.0, score), entry))
        if governed_ids:
            scored = [item for item in scored if item[1].cq_id in governed_rank]
            scored.sort(key=lambda item: (governed_rank[item[1].cq_id], -item[0]))
        else:
            scored.sort(key=lambda item: (-item[0], item[1].cq_id))
        return [
            CQMatch(
                cq_id=entry.cq_id,
                group=entry.group,
                question_en=entry.question_en,
                question_zh=entry.question_zh,
                semantic_path=entry.semantic_path,
                support_status=entry.support_status,
                score=round(score, 6),
                path_valid=validate_semantic_path(entry.semantic_path),
            )
            for score, entry in scored[:limit]
        ]

    def _bm25(self, query_tokens: list[str], document_index: int, k1: float = 1.5, b: float = 0.75) -> float:
        frequencies = self._term_frequencies[document_index]
        document_length = len(self._documents[document_index])
        score = 0.0
        for term in set(query_tokens):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            document_frequency = self._doc_frequency[term]
            inverse_document_frequency = math.log(
                1 + (len(self.entries) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * document_length / self._average_length)
            score += inverse_document_frequency * frequency * (k1 + 1) / denominator
        return score


def map_entity_mentions(query: str, aliases: dict[str, tuple[str, str]] | None = None) -> list[EntityPointer]:
    aliases = aliases or DEFAULT_ALIASES
    lowered = query.lower()
    pointers: list[EntityPointer] = []
    seen_uris: set[str] = set()
    for mention in sorted(aliases, key=len, reverse=True):
        if mention.lower() not in lowered:
            continue
        uri, label = aliases[mention]
        if uri in seen_uris:
            continue
        seen_uris.add(uri)
        pointers.append(
            EntityPointer(
                mention=mention,
                uri=uri,
                label=label,
                confidence=0.97,
                evidence="governed bilingual alias dictionary",
            )
        )
    return pointers
