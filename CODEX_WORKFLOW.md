# Working with the project agents

The main chat remains the lead for architecture, planning and integration.
AGENTS.md contains shared project instructions. The four reusable specialists live
in .codex/agents/:

| Agent | Use it for |
| --- | --- |
| data_curator | Recording plans, annotation readiness, coverage and session splits |
| training_evaluator | Model experiments, fitting, error analysis and evaluation |
| desktop_ui | PySide6 controls, labeler, overlay and desktop/audio integration |
| reviewer | Independent correctness review; no file edits |

These use the project-agent format in the [official Codex documentation](https://learn.chatgpt.com/docs/agent-configuration/subagents).
They omit model overrides so they inherit the main chat's model and reasoning
settings. Their instructions specialize the workflow; they do not train a model
or provide persistent shared conversation memory. Each assignment should include
the context and inputs it needs, and durable decisions belong in project docs.

## Start using them

Start a new Codex chat/session opened in this project so it can discover the new
project instructions and agent files. Client support and managed settings still
determine agent availability. The [AGENTS.md documentation](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
describes project instruction discovery. If the requested custom agents are not
available, report that explicitly rather than claiming they were used.

You can name an agent directly in an ordinary prompt. For example:

> Use data_curator to audit training/dataset.json and '1st test/dataset.json'.
> Keep the collections separate and do not edit them. Report confirmed tracks,
> reviewed coverage, class balance, session independence and the next data needed.

For independent audits:

> Delegate two read-only tasks in parallel: data_curator should assess dataset
> readiness; training_evaluator should check the training/evaluation code for
> leakage and misleading metrics. Wait for both, then give me a prioritized plan.

For implementation followed by review:

> Use desktop_ui to improve the AI suggestion review flow. Inspect the current
> behavior first, assign it the relevant UI files, and preserve saved labels and
> review scopes. After implementation and suitable visual checks, have reviewer
> inspect the result. Integrate fixes and report validation.

For a model experiment:

> Use training_evaluator to check whether the selected dataset is ready for a
> gunshot-only experiment. If ready, train to a separate path under artifacts/, tune
> on validation and report event metrics and latency. Keep test unused unless I
> explicitly request final evaluation. If data is missing, report what is needed.

AGENTS.md permits useful bounded delegation; small tasks can stay in the main
chat. Ask for parallel agents explicitly when you want a particular division of
work. The lead assigns one writer per file or artifact and handles shared schemas
and final integration. The reviewer advises without applying changes.

## Customize a role

Edit the corresponding .codex/agents/<name>.toml file. The name identifies the
agent, description explains when to select it, and developer_instructions defines
its standing workflow. Put the current assignment in your chat prompt.

Leave model settings absent to follow your main chat selection. Add model and
model_reasoning_effort only when you deliberately want a supported override.
Agent files inherit the parent's permissions; an instruction to avoid edits is a
behavioral constraint, not a separate read-only sandbox.

No project config.toml is needed for this setup with the documented default agent
settings. To customize global agent enablement or concurrency, consult the current
official documentation and your client's supported settings.
