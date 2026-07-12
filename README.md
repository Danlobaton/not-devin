# THIS IS **NOT DEVIN**

There are coding agents, there is Devin, and then there is **NOT DEVIN**.

NOT DEVIN is an eval-native coding agent: a system designed from the beginning to be dropped into different evaluation environments, measured, compared, and improved without rebuilding the agent around each harness.

That constraint creates room to push on the subsystems that make agentic software engineering actually work. The interesting problems are not confined to prompting: they live in the loop, the harness, the execution environment, the state model, and the feedback signals that distinguish progress from activity. This project is a place to make those decisions explicitly—and bake in a few opinions of my own.

## The Premise

Make a coding agent whose performance is easy to evaluate and whose behavior is possible to explain.

That requires a clean contract between the agent and the environment evaluating it. The runtime accepts a task, workspace, constraints, and available capabilities through stable interfaces. It emits structured events, artifacts, resource usage, and a terminal outcome. Provisioning, verification, and scoring stay outside the core agent loop.

Inside that boundary, the agent is treated as a runtime system rather than a chat interface with shell access. It has an explicit execution loop, typed tools, durable state, controlled side effects, and observable decisions. The model is an important component, but it is not the system.

## The Evaluation Contract

The same agent configuration should run against local fixtures, established coding benchmarks, regression suites, or custom internal tasks. Supporting a new environment should require an adapter, not a fork of the runtime.

An evaluation adapter should only need to define:

- How a task and workspace are provisioned.
- Which tools, resources, and time limits are available.
- How completion is verified and scored.
- Which traces, patches, costs, and timing metrics are collected.

This is not only benchmark plumbing. A portable evaluation boundary makes changes to prompts, models, tools, memory, and runtime policy comparable under controlled conditions—and makes performance claims reproducible rather than anecdotal.

## Opinions, Subject to Evidence

- The harness matters at least as much as the prompt.
- State should be explicit, replayable, and distinct from model-visible context.
- Tool calls are untrusted requests, not instructions from a privileged operator.
- Side effects need policy and identity; retries need idempotency.
- A plausible final message is not evidence that a software task was completed.
- Traces are part of the product when the system's behavior is probabilistic.
- Autonomy without evaluation is mostly confidence theater.
- The agent should adapt to an evaluation environment; the evaluation environment should not have to absorb the agent's internals.

These are starting positions, not sacred architecture. The point of the project is to implement them, break them, measure the results, and revise accordingly.

## Is This Devin?

No.

It is not affiliated with Devin or Cognition, does not attempt to reproduce their product, and should not be mistaken for either.

The name has been trying to tell you this the whole time.
