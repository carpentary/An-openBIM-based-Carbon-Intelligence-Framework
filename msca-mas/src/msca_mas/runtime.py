from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from .agents import AgentServices, build_agents
from .config import ProjectConfig
from .models import AgentName, SharedGlobalState
from .semantic_models import SharedGlobalSpace, TaskControlBlock


def _route(state: SharedGlobalState) -> str:
    pending = state.get("pending_agents", [])
    return pending[0] if pending else "__end__"


def _worker_node(agent: Any) -> Callable[[SharedGlobalState], dict[str, Any]]:
    def execute(state: SharedGlobalState) -> dict[str, Any]:
        pending = list(state.get("pending_agents", []))
        if not pending or pending[0] != agent.name.value:
            raise RuntimeError(f"scheduler expected {pending[0] if pending else None}, received {agent.name.value}")
        update = agent.run(state)
        remaining = pending[1:]
        update["pending_agents"] = remaining
        shared = SharedGlobalSpace.model_validate(
            update.get("shared_global_space") or state.get("shared_global_space") or {}
        )
        if shared.task_control_block is not None:
            completed = list(shared.task_control_block.completed_agents)
            if agent.name.value not in completed:
                completed.append(agent.name.value)
            control = shared.task_control_block.model_copy(
                update={
                    "pending_agents": remaining,
                    "current_agent": remaining[0] if remaining else None,
                    "completed_agents": completed,
                }
            )
            shared = shared.model_copy(update={"task_control_block": control})
            update["shared_global_space"] = shared.model_dump(mode="json")
        return update

    execute.__name__ = f"run_{agent.name.value}"
    return execute


def build_workflow(
    config: ProjectConfig | None = None,
    services: AgentServices | None = None,
):
    config = config or ProjectConfig.load()
    services = services or AgentServices.create(config)
    agents = build_agents(services)
    builder = StateGraph(SharedGlobalState)
    builder.add_node(AgentName.ORCHESTRATOR.value, agents[AgentName.ORCHESTRATOR.value].run)
    for name in (
        AgentName.METADATA,
        AgentName.EXTRACTOR,
        AgentName.READER,
        AgentName.CALCULATOR,
        AgentName.SYNTHESIS,
    ):
        builder.add_node(name.value, _worker_node(agents[name.value]))
    builder.add_edge(START, AgentName.ORCHESTRATOR.value)
    route_map = {
        AgentName.METADATA.value: AgentName.METADATA.value,
        AgentName.EXTRACTOR.value: AgentName.EXTRACTOR.value,
        AgentName.READER.value: AgentName.READER.value,
        AgentName.CALCULATOR.value: AgentName.CALCULATOR.value,
        AgentName.SYNTHESIS.value: AgentName.SYNTHESIS.value,
        "__end__": END,
    }
    for name in (
        AgentName.ORCHESTRATOR,
        AgentName.METADATA,
        AgentName.EXTRACTOR,
        AgentName.READER,
        AgentName.CALCULATOR,
        AgentName.SYNTHESIS,
    ):
        builder.add_conditional_edges(name.value, _route, route_map)
    return builder.compile()


def invoke_query(query: str, config: ProjectConfig | None = None, services: AgentServices | None = None) -> dict[str, Any]:
    workflow = build_workflow(config=config, services=services)
    return workflow.invoke({"query": query})
