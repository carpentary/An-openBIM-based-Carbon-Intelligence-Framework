"""Indexed forward chaining for positive class and object property rules.

Each inferred triple is recorded with its source rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Set

from rdflib import BNode, Graph, Namespace, RDF, RDFS, URIRef

SWRL = Namespace("http://www.w3.org/2003/11/swrl#")
Term = URIRef | BNode
Binding = dict[Term, Term]


@dataclass
class Atom:
    kind: str
    predicate: URIRef
    arg1: Term
    arg2: Optional[Term] = None


@dataclass
class Rule:
    uri: URIRef
    body: list[Atom]
    head: list[Atom]


@dataclass
class Inference:
    triple: tuple[Any, Any, Any]
    rule: str


class IndexedSWRLForwardChainer:
    """Apply supported rules using an index of class memberships."""

    def __init__(self, data: Graph, rules: Graph):
        self.data = data
        self.rules_graph = rules
        self.variables = set(rules.subjects(RDF.type, SWRL.Variable))
        self.rules = self._parse_rules()
        self._subclass_cache: dict[URIRef, set[URIRef]] = {}
        self._type_index: dict[URIRef, set[Term]] = {}
        self._instance_cache: dict[URIRef, tuple[set[Term], int]] = {}
        for subject, _, cls in data.triples((None, RDF.type, None)):
            if isinstance(cls, URIRef):
                self._type_index.setdefault(cls, set()).add(subject)
        self.inferences: list[Inference] = []

    def _list(self, head):
        out = []
        cur = head
        seen = set()
        while cur and cur != RDF.nil and cur not in seen:
            seen.add(cur)
            first = self.rules_graph.value(cur, RDF.first)
            if first is None:
                break
            out.append(first)
            cur = self.rules_graph.value(cur, RDF.rest)
        return out

    def _atom(self, node) -> Atom:
        types = set(self.rules_graph.objects(node, RDF.type))
        if SWRL.ClassAtom in types:
            return Atom(
                "class",
                self.rules_graph.value(node, SWRL.classPredicate),
                self.rules_graph.value(node, SWRL.argument1),
            )
        if SWRL.IndividualPropertyAtom in types:
            return Atom(
                "object",
                self.rules_graph.value(node, SWRL.propertyPredicate),
                self.rules_graph.value(node, SWRL.argument1),
                self.rules_graph.value(node, SWRL.argument2),
            )
        raise ValueError(f"Unsupported atom {node}: {types}")

    def _parse_rules(self) -> list[Rule]:
        out = []
        for r in sorted(set(self.rules_graph.subjects(RDF.type, SWRL.Imp)), key=str):
            body = self.rules_graph.value(r, SWRL.body)
            head = self.rules_graph.value(r, SWRL.head)
            out.append(Rule(r, [self._atom(x) for x in self._list(body)], [self._atom(x) for x in self._list(head)]))
        return out

    def _resolve(self, t: Term, b: Binding) -> Optional[Term]:
        return b.get(t) if t in self.variables else t

    def _superclasses(self, cls: URIRef) -> set[URIRef]:
        cached = self._subclass_cache.get(cls)
        if cached is not None:
            return cached
        out: set[URIRef] = set()
        stack = [cls]
        while stack:
            c = stack.pop()
            for sup in self.data.objects(c, RDFS.subClassOf):
                if isinstance(sup, URIRef) and sup not in out:
                    out.add(sup)
                    stack.append(sup)
        self._subclass_cache[cls] = out
        return out

    def _is_instance_of(self, individual: Term, cls: URIRef) -> bool:
        for asserted in self.data.objects(individual, RDF.type):
            if asserted == cls or cls in self._superclasses(asserted):
                return True
        return False

    def _instances_of(self, cls: URIRef) -> set[Term]:
        """All current instances of cls (asserted, inferred, subclass closure)."""

        cached, cached_size = self._instance_cache.get(cls, (set(), -1))
        current_size = sum(len(s) for s in self._type_index.values())
        if cached_size == current_size:
            return cached
        instances: set[Term] = set()
        for asserted_cls, subjects in self._type_index.items():
            if asserted_cls == cls or cls in self._superclasses(asserted_cls):
                instances |= subjects
        self._instance_cache[cls] = (instances, current_size)
        return instances

    def _extend_class(self, atom: Atom, b: Binding) -> Iterable[Binding]:
        val = self._resolve(atom.arg1, b)
        if val is not None:
            if self._is_instance_of(val, atom.predicate):
                yield dict(b)
            return
        for subject in sorted(self._instances_of(atom.predicate), key=str):
            nb = dict(b)
            nb[atom.arg1] = subject
            yield nb

    def _extend_object(self, atom: Atom, b: Binding) -> Iterable[Binding]:
        s = self._resolve(atom.arg1, b)
        o = self._resolve(atom.arg2, b)
        for ts, _, to in self.data.triples((s, atom.predicate, o)):
            nb = dict(b)
            if atom.arg1 in self.variables:
                old = nb.get(atom.arg1)
                if old is not None and old != ts:
                    continue
                nb[atom.arg1] = ts
            if atom.arg2 in self.variables:
                old = nb.get(atom.arg2)
                if old is not None and old != to:
                    continue
                nb[atom.arg2] = to
            yield nb

    def matches(self, rule: Rule) -> list[Binding]:
        bindings: list[Binding] = [{}]
        for atom in rule.body:
            nxt: list[Binding] = []
            for b in bindings:
                nxt.extend(self._extend_class(atom, b) if atom.kind == "class" else self._extend_object(atom, b))
            bindings = nxt
            if not bindings:
                break
        return bindings

    def apply(self, rule: Rule, b: Binding) -> int:
        added = 0
        for atom in rule.head:
            s = self._resolve(atom.arg1, b)
            if s is None:
                raise ValueError(f"Unbound head subject in {rule.uri}")
            if atom.kind == "class":
                triple = (s, RDF.type, atom.predicate)
            else:
                o = self._resolve(atom.arg2, b)
                if o is None:
                    raise ValueError(f"Unbound head object in {rule.uri}")
                triple = (s, atom.predicate, o)
            if triple not in self.data:
                self.data.add(triple)
                added += 1
                self.inferences.append(Inference(triple, str(rule.uri)))
                if atom.kind == "class" and isinstance(atom.predicate, URIRef):
                    self._type_index.setdefault(atom.predicate, set()).add(s)
                    self._instance_cache.clear()
        return added

    def run(self, max_iterations: int = 20) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "iterations": 0,
            "new_triples": 0,
            "rule_firings": 0,
            "per_rule": {},
        }
        for _ in range(max_iterations):
            stats["iterations"] += 1
            iteration_added = 0
            for rule in self.rules:
                r_added = 0
                for b in self.matches(rule):
                    n = self.apply(rule, b)
                    if n:
                        stats["rule_firings"] += 1
                        r_added += n
                if r_added:
                    stats["per_rule"][str(rule.uri)] = stats["per_rule"].get(str(rule.uri), 0) + r_added
                iteration_added += r_added
            stats["new_triples"] += iteration_added
            if iteration_added == 0:
                return stats
        raise RuntimeError(f"Forward chaining did not converge in {max_iterations} iterations")
