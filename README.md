# semantic-layer-lite

A tiny semantic layer for SQLite: you approve the joins and metrics once, then every question an agent asks compiles to one safe query.

![License: MIT](https://img.shields.io/badge/license-MIT-0A2942) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-0A2942) ![Dependencies: none](https://img.shields.io/badge/dependencies-none-0A2942)

![A SQLite file goes into infer, which proposes joins and metrics; a person approves or rejects them in semantic.json; every agent question then compiles through only the approved joins into one read-only query, and each answer comes back with its join path and metric definition.](docs/diagram.svg)

## S: Situation

Picture a kitchen where ten cooks each make "the house soup" from memory, so it tastes different every time and nobody can say which cook was right. A recipe card on the wall fixes that. Everyone cooks from the same card.

A company database is that kitchen. Revenue sits in an order lines table, but the same orders also link to payments and to shipments, and one order can have several of each. A person who knows the schema picks the right route. An AI agent writing SQL on its own sees several routes that all look reasonable, joins along one of them, and returns a total that is quietly too big or too small. The query runs, the number looks plausible, and nothing flags it. When a join repeats rows like this, the mistake is called a fan-out, and it is the most common way an agent's revenue figure goes wrong without an error.

## T: Task

Think of the recipe card again. The job is to write the card once, have a person check it, and make every cook follow it, including the cook who is a program.

In technical terms, this repo must do five things, and do them with the Python standard library only, offline, and with seeded, repeatable output:

1. Read a SQLite file and propose its joins, metrics and dimensions (a dimension is a column you group or filter by, such as region or month) for a person to review.
2. Turn a metric request into exactly one read-only SELECT that uses only joins a person approved.
3. Refuse a request instead of guessing when the route is missing, when two routes are equally short, or when a join would repeat rows and inflate a sum.
4. Show the join path and the metric definition behind every number.
5. Prove all of it on a seeded demo database with known true joins, using a bench that anyone can rerun.

## A: Action

The recipe card is a file called `semantic.json`. The program writes a first draft of the card, a person strikes out what is wrong, and from then on every question is cooked from the card.

How it works, step by step:

1. **infer** opens the database read-only and lists tables, columns, types, primary keys and declared foreign keys. Declared keys are taken as written. Undeclared joins are proposed only when three things hold at once: the column name points at the target table (`product_id` to `products`), the target column is unique, and every non-null value in the source column exists in the target. That last check runs as SQL and is capped. Two integer columns that merely overlap do not become a join. Each proposed join records its direction (many rows point at one) and the evidence behind it. Numeric non-key columns become proposed measures, ISO date text becomes time dimensions, and low-cardinality text becomes categorical dimensions. Everything is written with `"status": "proposed"`. Keys are read when the schema declares them, but the approved model, not the schema, decides which joins are ever used. This follows the idea from Delphi-Data's benchmark walkthrough of keeping keys in the semantic layer rather than in the database definition.
2. **approve and reject** are one-word edits. Change `"proposed"` to `"approved"` or `"rejected"` in the file by hand, or run `approve --all`, `approve --join ID`, `reject --join ID`. Nothing proposed is usable until it is approved.
3. **compile** builds a small plan first: the base table of the metric, a route to every other table the query touches, the time grain and the filters. The SQL string is rendered from that plan and nothing else. This plan-first design is the idea we took from dbt-labs/metricflow. Routes come from a breadth-first search over approved joins only. The compiler refuses when there is no route, and when two routes are equally short (a deliberate difference from tools that pick one automatically, because an ambiguous route is the exact trap this repo is about). It also refuses when a join would repeat base-table rows for a metric that adds up, such as a sum. Counting distinct orders is allowed across such a join because repeats do not change it. Filter values become `?` parameters and are never pasted into the SQL. A row limit always applies. Time grain (day, month, year) is a normal request parameter, as in metricflow.
4. **run** executes the statement on a connection opened read-only, with `PRAGMA query_only = ON` set as a second lock.
5. **explain** prints the same plan the SQL was built from: the metric definition, each join id with its direction and whether it fans out, and the grain. The `query --json` output attaches the join ids and metric definition to every result row, an audit-trail idea from cloudbadal007/enterprise-ontology-ai-agents. Cube's central promise, one shared definition per metric that every tool and agent must use, frames the whole repo. Where Cube hides the plumbing, this repo shows the join path on purpose.
6. **bench** scores 20 fixed questions against hand-written reference SQL and prints the compiled SQL beside each result with `--verbose`. Each question runs on a fresh connection so no result depends on order. The scoring-against-a-fixed-question-set idea comes from the two LLM semantic-layer benchmark repos. This repo's default path has no model in it.
7. **ask --live** is optional. It sends a plain question to a chat endpoint you name, using a key from your own environment variable, and gets back a suggested metric query. That suggestion still goes through the compiler, so it can use approved things only. No test and no bench uses it.

## R: Result

The bench measures how many join paths the raw schema offers from orders to the revenue column and how many of them disagree with the approved route, how well join inference matches the generator's true joins, and how many of the 20 compiled questions match their reference SQL.

<!-- RESULT:START -->
| Join inference | True joins | Proposed | Correct | Precision | Recall | Declared found | Inferred found |
|---|---|---|---|---|---|---|---|
| demo database | 6 | 6 | 6 | 1.000 | 1.000 | 2 | 4 |

| Questions | Compiled and run | Exact match with reference SQL | Used approved joins only |
|---|---|---|---|
| 20 | 20 | 20 | 20 |

| Route from orders to the revenue column | Grand total | Same monthly answer as the approved route |
|---|---|---|
| approved route (compiled) | 191233.49 | yes |
| orders -> order_items | 191233.49 | yes |
| orders -> shipments -> order_items | 199089.83 | no |

The raw schema offers 2 distinct simple join paths of up to 4 hops from orders to the revenue column, and 1 of them gives a different monthly revenue than the approved route.
Join inference proposed 6 joins against the generator's 6 true joins, with precision 1.000 and recall 1.000.
Compiled SQL matched hand-written reference SQL on 20 of 20 questions, and 20 of 20 used approved joins only.

Measured with python3 bench/run_bench.py, Python 3.12.13, arm64.
<!-- RESULT:END -->

## Quickstart

```bash
git clone https://github.com/wilsonwu-ai/semantic-layer-lite
cd semantic-layer-lite
python3 -m unittest discover -s tests
python3 bench/run_bench.py
```

Build the demo database, let infer propose a draft, and approve it:

```bash
python3 -m semantic_layer_lite demo-db demo.sqlite
python3 -m semantic_layer_lite infer demo.sqlite -o proposed.json
python3 -m semantic_layer_lite approve proposed.json --all
```

Ask for revenue by month through the reviewed model in `examples/`, and see the route behind it:

```bash
python3 -m semantic_layer_lite query demo.sqlite examples/semantic.json --metric revenue --by month
python3 -m semantic_layer_lite explain examples/semantic.json --metric revenue --by region
```

Watch it refuse the fan-out trap (revenue split by shipping carrier would count some lines twice):

```bash
python3 -m semantic_layer_lite compile examples/semantic.json --metric revenue --by carrier
```

To try it on your own data, run `infer` on a copy of one SQLite file you own, open the `semantic.json` it writes, read the joins it proposes, and change the status of any join you disagree with from `"proposed"` to `"rejected"`, or run `reject semantic.json --join ID`. Then approve the rest and query.

### Commands

| Command | What it does |
|---|---|
| `infer DB -o semantic.json` | Propose joins, metrics and dimensions. All start as proposed. |
| `approve MODEL --all`, `--join ID`, `--metric NAME` | Mark items approved. |
| `reject MODEL --join ID`, `--metric NAME` | Mark items rejected. |
| `compile MODEL --metric M --by D` | Print the one SELECT and its parameters. Add `--json` for the plan. |
| `query DB MODEL --metric M --by D` | Compile, run read-only, print rows plus the explanation. |
| `explain MODEL --metric M --by D` | Print the join path and metric definition only. |
| `demo-db OUT` | Write the seeded six-table store database. |
| `ask DB MODEL "question" --live --endpoint URL --llm NAME` | Optional language-model mapping, never used by tests. |

Query options: `--by` takes a dimension name, `name:grain` (for example `order_date:month`) or a bare grain (`day`, `month`, `year`) that applies to the metric's time dimension. `--filter` takes `region=West`, `order_date>=2026-03-01` or `'region in East,West'`. `--limit` sets the row limit (default 1000).

### The semantic.json file

| Part | What it holds |
|---|---|
| `entities` | Each table with its primary key and declared column types. |
| `joins` | `id`, the two tables and columns, `cardinality` (`many_to_one` or `one_to_one`), `status` (`proposed`, `approved`, `rejected`) and `evidence`. |
| `metrics` | `name`, `base_table`, `aggregation` (`sum`, `count`, `avg`, `min`, `max`, `count_distinct`), `expression` (columns of the base table with `+ - * / ( )` and numbers, or `*` for `count`), `status`, optional `time_dimension`. |
| `dimensions` | `name`, `table`, `column`, `kind` (`time` or `categorical`). |

Column names inside an expression are checked against the base table and quoted by the compiler. Anything else in an expression, such as a function call or a semicolon, fails validation when the file is loaded.

## How it works

| File | Job |
|---|---|
| `semantic_layer_lite/infer.py` | Read the database, propose joins, measures and dimensions. |
| `semantic_layer_lite/model.py` | Load, validate, edit and save `semantic.json`. |
| `semantic_layer_lite/compile.py` | Plan, route search, refusals, SQL rendering. |
| `semantic_layer_lite/run.py` | Read-only execution and row comparison. |
| `semantic_layer_lite/explain.py` | Plan and result explanations. |
| `semantic_layer_lite/datasets.py` | Seeded six-table store with its true joins. |
| `semantic_layer_lite/live.py` | Optional language-model mapping, behind `--live`. |
| `semantic_layer_lite/cli.py` | The command line. |
| `bench/run_bench.py`, `bench/questions.json` | The bench and its 20 questions with reference SQL. |
| `examples/semantic.json` | The reviewed model for the demo database. |

## Limits

- It is read-only by design, and it is not an access-control layer. It does not hide tables from anyone who can open the SQLite file, and the approved model is a guide for the compiler, not a permission system. Do not expose it on a network. Use file permissions and a copy of the data for anything sensitive.
- `semantic.json` is trusted input. Expressions are validated and quoted, but a person who can edit the file decides what the metrics mean.
- Inference is a heuristic. It can miss a join whose columns do not follow the naming patterns, and it can propose a join that is real in the data but wrong for your business. That is why a person reviews it. Only single-column joins are supported, and the check covers at most a capped number of distinct values per column.
- Time columns must hold ISO date or datetime text. Time filters compare text, so write dates as `YYYY-MM-DD`.
- Joins from the base table are `LEFT JOIN`, so base rows with no match show up as a NULL group instead of disappearing.
- The fan-out rule only looks at the joins a query uses. It treats `sum`, `count` and `avg` as unsafe across a one-to-many step, and `count_distinct`, `min` and `max` as safe. All metrics in one query must share a base table, so there is no multi-fact planning, no ratio or cumulative metric, and no window functions.
- The bench uses a synthetic store database whose true joins the generator knows. It shows the failure mode and that the compiler avoids it. It does not show how inference performs on your schema.
- The optional `--live` mode was written against the common chat-completions request format and has not been exercised by the tests or the bench.
- Written for SQLite only, with no server and no cross-database support.

## Built on ideas from

| Repo | Licence | Idea we took | Code copied |
|---|---|---|---|
| [dbt-labs/metricflow](https://github.com/dbt-labs/metricflow) | Apache-2.0 | Build an inspectable plan for each request first, render the SQL from it, and treat time grain as a normal request parameter. | No |
| [cube-js/cube](https://github.com/cube-js/cube) | NOASSERTION (ideas only) | Define each metric once so every consumer goes through it, and state plainly what a convenience path does and does not protect. | No |
| [dbt-labs/semantic-layer-llm-benchmarking](https://github.com/dbt-labs/semantic-layer-llm-benchmarking) | Apache-2.0 | Benchmark answers given through the semantic layer, and ship the files needed to rerun it and get the same results. | No |
| [Delphi-Data/delphi-semantic-layer-llm-benchmark](https://github.com/Delphi-Data/delphi-semantic-layer-llm-benchmark) | MIT | Keep keys in the semantic layer rather than the database definition. | No |
| [cloudbadal007/enterprise-ontology-ai-agents](https://github.com/cloudbadal007/enterprise-ontology-ai-agents) | MIT | Promise a full audit trail on every answer; here that means the join ids and metric definition on each result row. | No |

## License

MIT, see [LICENSE](LICENSE).
