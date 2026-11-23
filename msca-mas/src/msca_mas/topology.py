from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol


Point3D = tuple[float, float, float]


@dataclass(frozen=True)
class AABB:
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float

    def __post_init__(self) -> None:
        if self.xmin > self.xmax or self.ymin > self.ymax or self.zmin > self.zmax:
            raise ValueError("AABB minima must not exceed maxima")

    @property
    def centroid(self) -> Point3D:
        return (
            (self.xmin + self.xmax) / 2,
            (self.ymin + self.ymax) / 2,
            (self.zmin + self.zmax) / 2,
        )

    def expanded(self, epsilon: float) -> "AABB":
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        return AABB(
            self.xmin - epsilon,
            self.ymin - epsilon,
            self.zmin - epsilon,
            self.xmax + epsilon,
            self.ymax + epsilon,
            self.zmax + epsilon,
        )

    def intersects(self, other: "AABB") -> bool:
        return not (
            self.xmax < other.xmin
            or other.xmax < self.xmin
            or self.ymax < other.ymin
            or other.ymax < self.ymin
            or self.zmax < other.zmin
            or other.zmax < self.zmin
        )

    def footprint_overlap_area(self, other: "AABB") -> float:
        width = max(0.0, min(self.xmax, other.xmax) - max(self.xmin, other.xmin))
        height = max(0.0, min(self.ymax, other.ymax) - max(self.ymin, other.ymin))
        return width * height

    def separation(self, other: "AABB") -> float:
        dx = max(self.xmin - other.xmax, other.xmin - self.xmax, 0.0)
        dy = max(self.ymin - other.ymax, other.ymin - self.ymax, 0.0)
        dz = max(self.zmin - other.zmax, other.zmin - self.zmax, 0.0)
        return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass(frozen=True)
class GeometryEntity:
    entity_id: str
    bounds: AABB


@dataclass(frozen=True)
class Polyhedron:
    vertices: tuple[Point3D, ...]
    triangular_faces: tuple[tuple[int, int, int], ...]

    def contains(self, point: Point3D, epsilon: float = 1e-9) -> bool:
        # Use a skew ray to reduce edge and vertex intersections.
        direction = _normalise((1.0, 0.271828, 0.141421))
        intersections = 0
        for face in self.triangular_faces:
            triangle = tuple(self.vertices[index] for index in face)
            distance = _ray_triangle_distance(point, direction, triangle, epsilon)
            if distance is not None and distance > epsilon:
                intersections += 1
        return intersections % 2 == 1


class CandidateIndex(Protocol):
    backend: str

    def query(self, bounds: AABB) -> Iterable[GeometryEntity]: ...


class BruteForceAABBIndex:
    """Deterministic fallback when the optional native R-tree is unavailable."""

    backend = "bruteforce_aabb"

    def __init__(self, entities: Iterable[GeometryEntity]):
        self._entities = tuple(entities)

    def query(self, bounds: AABB) -> Iterable[GeometryEntity]:
        return (entity for entity in self._entities if entity.bounds.intersects(bounds))


class RTreeAABBIndex:
    backend = "rtree"

    def __init__(self, entities: Iterable[GeometryEntity]):
        try:
            from rtree.index import Index, Property  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install the optional 'rtree' package to use the R-tree backend") from exc
        self._entities = tuple(entities)
        properties = Property()
        properties.dimension = 3
        self._index = Index(properties=properties)
        for position, entity in enumerate(self._entities):
            box = entity.bounds
            self._index.insert(position, (box.xmin, box.ymin, box.zmin, box.xmax, box.ymax, box.zmax))

    def query(self, bounds: AABB) -> Iterable[GeometryEntity]:
        positions = self._index.intersection(
            (bounds.xmin, bounds.ymin, bounds.zmin, bounds.xmax, bounds.ymax, bounds.zmax)
        )
        return (self._entities[position] for position in positions)


@dataclass(frozen=True)
class TopologyRelation:
    target_id: str
    candidate_id: str
    is_connected: bool
    is_below: bool
    is_inside_excavation: bool | None
    index_backend: str


class TwoStageTopologyResolver:
    """Appendix-B topology contract: candidate index, then exact predicates."""

    def __init__(self, entities: Iterable[GeometryEntity], prefer_rtree: bool = True):
        materialised = tuple(entities)
        if prefer_rtree:
            try:
                self.index: CandidateIndex = RTreeAABBIndex(materialised)
            except RuntimeError:
                self.index = BruteForceAABBIndex(materialised)
        else:
            self.index = BruteForceAABBIndex(materialised)

    def resolve(
        self,
        target: GeometryEntity,
        excavation: Polyhedron | None = None,
        epsilon: float | None = None,
    ) -> list[TopologyRelation]:
        if epsilon is None:
            raise ValueError("epsilon must be explicit and expressed in the model coordinate unit")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        candidates = self.index.query(target.bounds.expanded(epsilon))
        relations: list[TopologyRelation] = []
        for candidate in candidates:
            if candidate.entity_id == target.entity_id:
                continue
            connected = target.bounds.separation(candidate.bounds) <= epsilon
            below = (
                candidate.bounds.zmax <= target.bounds.zmin + epsilon
                and candidate.bounds.footprint_overlap_area(target.bounds) > 0
            )
            inside = excavation.contains(candidate.bounds.centroid, epsilon=max(epsilon * 1e-3, 1e-9)) if excavation else None
            relations.append(
                TopologyRelation(
                    target_id=target.entity_id,
                    candidate_id=candidate.entity_id,
                    is_connected=connected,
                    is_below=below,
                    is_inside_excavation=inside,
                    index_backend=self.index.backend,
                )
            )
        return relations


def _normalise(vector: Point3D) -> Point3D:
    length = math.sqrt(sum(component * component for component in vector))
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def _subtract(left: Point3D, right: Point3D) -> Point3D:
    return left[0] - right[0], left[1] - right[1], left[2] - right[2]


def _cross(left: Point3D, right: Point3D) -> Point3D:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: Point3D, right: Point3D) -> float:
    return sum(a * b for a, b in zip(left, right))


def _ray_triangle_distance(
    origin: Point3D,
    direction: Point3D,
    triangle: tuple[Point3D, Point3D, Point3D],
    epsilon: float,
) -> float | None:
    vertex0, vertex1, vertex2 = triangle
    edge1 = _subtract(vertex1, vertex0)
    edge2 = _subtract(vertex2, vertex0)
    h = _cross(direction, edge2)
    determinant = _dot(edge1, h)
    if -epsilon < determinant < epsilon:
        return None
    inverse = 1.0 / determinant
    s = _subtract(origin, vertex0)
    u = inverse * _dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = _cross(s, edge1)
    v = inverse * _dot(direction, q)
    if v < 0.0 or u + v > 1.0:
        return None
    distance = inverse * _dot(edge2, q)
    return distance if distance > epsilon else None
