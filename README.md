# THIS IS **NOT DEVIN**

There are coding agents, there is Devin, and then there is NOT DEVIN.

NOT DEVIN is an eval-native coding agent that takes a GitHub issue, inspects a repository, produces a validated patch, and explains how it got there.

The project is designed around a simple premise: coding agents should be easy to evaluate, debug, compare, and improve without rebuilding them around every new benchmark or execution harness.

## What It Does

Given a GitHub issue and a repository, NOT DEVIN:

1. Reads the issue.
2. Inspects the codebase.
3. Identifies relevant files.
4. Edits the implementation.
5. Runs tests and linting.
6. Iterates on failures.
7. Produces a patch and implementation summary.
8. Opens a draft pull request.
9. Generates a PR ready description

## Why This Project Exists

The interesting problems in coding agents are not limited to prompting. They also live in:

* the agent loop
* tool execution
* state management
* verification
* observability
* failure recovery
* evaluation

NOT DEVIN treats the model as one component inside a larger runtime.

The model proposes actions. The runtime validates and executes them.

## Evaluation-First Design

The core agent is kept separate from repository provisioning, hidden tests, scoring, and benchmark-specific logic.

This makes it possible to run the same agent across:

* local demo repositories
* custom GitHub issue sets
* coding-agent benchmarks
* different evaluation harnesses

Each run produces both a result and a trace of how that result was reached.

## Agent Capabilities

The initial tool surface includes:

```
read_issue
search_code
read_file
write_file
run_tests
run_linter
git_diff
git_status
```

The agent does not receive unrestricted shell access.

## Example Run

```
Issue received
→ repository inspected
→ relevant code located
→ implementation edited
→ tests executed
→ failure analyzed
→ patch revised
→ verification passes
→ patch and summary produced
```

A demo repository can include issues such as:

* incorrect pagination
* missing input validation
* broken date sorting
* uncovered edge cases
* stale cache behavior

## Initial Scope

The first version should:

* accept a GitHub issue
* operate on a local repository
* use controlled tools
* modify code
* run verification
* generate a patch
* emit an execution trace
* produce a PR-ready summary

It does not need:

* a web interface
* autonomous issue selection
* unrestricted terminal access
* distributed scheduling
* automatic PR publication

## Project Goals

NOT DEVIN is a place to explore:

* how issue text becomes an executable task
* how agents recover from failed hypotheses
* how progress is distinguished from activity
* how coding-agent runs can be evaluated
* how agent behavior can be made explainable

## Is This Devin?

No.

NOT DEVIN is not affiliated with Devin or Cognition. It does not attempt to reproduce their product and should not be mistaken for either.

The name has been trying to tell you this the whole time.
