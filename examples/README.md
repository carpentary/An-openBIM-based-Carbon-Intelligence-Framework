# Representative reproducibility materials

This directory provides a small executable example and an anonymized query
excerpt. They serve different purposes and must not be combined into a station
result or a benchmark performance claim.

## Files

| Material | Contents and scope |
| --- | --- |
| `ontology/schema.ttl` | A subset of the project ontology covering the S-E-P-R-C chain and the vocabulary needed by the included rules. It is not the entire ontology. |
| `ontology/rules.ttl` | Three rules selected from the semantic rule examples: resolve lowest slab position, classify a structural bottom slab, and inherit a process family from a template. |
| `ontology/shapes.ttl`, `total_by_stage.rq` | Small validation and stage-aggregation checks for the synthetic graph. |
| `ifc/station_fragment.ifc` | A newly authored synthetic IFC4 fragment with one rectangular concrete slab, metre units and quantity/property sets. It contains no original model geometry or identifiers. |
| `reference/` | Two illustrative process entries, three resource-quota lines, three emission factor records, and declared geometry and delivery assumptions. All numbers in these files are synthetic, not copied from quota books or factor databases. |
| `expected_outputs.json` | Analytical expectations for the synthetic example, not measured emissions or study results. |
| `benchmark/` | Fifteen anonymized excerpts from the study's query/ground-truth collection, with aggregate reference inputs. Each mode has three normal requests and two traps. These references are not the outputs of the synthetic IFC model. |

External vocabulary terms such as QUDT, PROV-O, OWL and SWRL are referenced by
IRI. Their source databases and documentation are not mirrored.

## Run the synthetic example

From the repository root:

```sh
cd msca-mas
uv sync --locked --python 3.11 --extra dev --extra validation
uv run --no-sync python ../examples/run_example.py
uv run --no-sync python ../examples/check_query_references.py
```

The first script reuses the released IFC property reader, SWRL chainer and
transport calculator. It reads the rectangular extrusion, checks its declared
volume, classifies the slab, applies the illustrative CPD/EFD records, and
validates and aggregates the resulting RDF graph. Outputs are written under
`examples/output/`. It needs neither an API key nor model downloads.

The geometry example is deliberately restricted to a metre-based,
axis-aligned rectangular extrusion and a two-layer rectangular excavation.
It does not reproduce geological meshing, arbitrary foundation-pit
reconstruction, learned matching, or an entire metro station. It is not a
replacement for unreleased production pipeline stages.

The second script checks the suite structure and independently recomputes the
published numerical references from the supplied aggregate inputs. For
corrections it checks the proposed field and target representation. It does
not interpret queries with an LLM, run the complete MAS, or measure trap
interception accuracy.

## Query excerpt

`benchmark/queries.csv` contains the questions; `queries_and_ground_truth.json`
includes expected intent, status, issue codes, assertions, units and tolerances.
The selection covers aggregation, ranking, transport, sensitivity, correction
previews, assessment scope, and conflicting instructions. Cases were selected
for task coverage, not because a model passed them.

Station names, project RDF namespaces, and case/assertion identifiers were
replaced. Expected numerical answers, rejection reasons and tolerances were
retained. The aggregate inputs are the reference snapshot for these selected
queries, not the eight-station results table in the paper. No new human
annotation exercise or agreement statistic is claimed.
Assertions about a signed logistics note describe an evaluation prompt; no
original note is provided as evidence of a real-world correction.

The complete 150-query collection, full project graph, hosted model responses,
and original station inputs are not included. This release supports inspection
and execution of the representative materials, not reproduction of the full
reported experiment. The example schema does not supply the full graph required
by these project-level questions.
