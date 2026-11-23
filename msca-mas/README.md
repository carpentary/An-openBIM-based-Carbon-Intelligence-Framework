# MSCA-MAS

MSCA-MAS is an ontology-constrained multi-agent system for embodied carbon
assessment in metro station construction. It answers queries against a carbon
knowledge graph (CKG), evaluates temporary scenarios, and prepares corrections
for review. An approved correction is written as a separate RDF delta; the
source graph is not edited in place.

See the accompanying paper for the evaluation and its results. This repository
contains the source code, not the project data, query sets, model weights,
or results.

## Install

With Python 3.11 and uv, run these commands from the `msca-mas/` directory:

```sh
uv sync --locked --python 3.11 --extra dev --extra validation
uv run --no-sync msca-mas --help
uv run --no-sync pytest
```

The unit tests do not need project files or an API key.

## Configure and run

Copy `config/project.toml` and set the paths to your ontology, CKG release, CPD,
EFD, and project reference inputs. Paths are relative to the configuration
file. Keep the CKG release layout and set the summary, shard pattern, and
domain graph paths to match that release. The preparation commands also require
the IFC models and topology inputs. Keep local copies of these inputs and any
generated outputs out of Git.

```sh
uv run --no-sync msca-mas --config config/project.toml doctor
uv run --no-sync msca-mas --config config/project.toml ask --no-auto-index --json "What are the project carbon totals?"
```

`doctor` lists missing inputs. Use `prepare` to build an index and prepare
project records. For a study rerun, use the same input snapshots, configuration,
and index as the evaluated run.

The planner runs without a hosted model by default. For DeepSeek planning, set
`MSCA_LLM_PROVIDER=deepseek`, `DEEPSEEK_MODEL=deepseek-v4-flash`, and
`DEEPSEEK_THINKING=disabled`. Supply `DEEPSEEK_API_KEY` through your environment.
The `retrieval` extra enables local BGE models at the revisions pinned in the
configuration. Without those models, `auto` uses deterministic retrieval.

## Evaluate a query suite

Supply the query suite as JSON. The commands write predictions and scores
locally. Scoring existing predictions does not call a language model.

```sh
uv run --no-sync msca-mas --config config/project.toml gt-run --dataset data/evaluation/queries.json --output-dir outputs/evaluation
uv run --no-sync msca-mas gt-score --dataset data/evaluation/queries.json --predictions outputs/evaluation/predictions_en.json --output outputs/evaluation/scores.json
```

## Source layout

- `runtime.py`, `agents.py`, and `query_contract.py` plan and run tasks.
- `ontology_profiles.py` and the retrieval modules link queries to CKG and
  reference records.
- `mas_program.py` and `calculations.py` build and execute calculation steps.
- `ontology_runtime.py` and `transactions.py` check the graph and handle
  correction previews and deltas.
- `evaluation.py` and `evaluation_runner.py` run and score a supplied query suite.
