from __future__ import annotations

import json
from typing import Any, Dict, Optional

from rl.reward_logic import (
    DEFAULT_STOP_ACTION,
    INVALID_ACTION_REWARD,
    replay_episode,
)
from verl.interactions.sequential_triple_selection_interaction import SequentialTripleSelectionInteraction


class WebQSPRetrieverInteraction(SequentialTripleSelectionInteraction):
    """WebQSP retriever interaction with STOP-only stop reward."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
        initial_observation: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        if ground_truth is None or initial_observation is None:
            raise ValueError("ground_truth and initial_observation are required for webqsp retriever interaction")
        return await super().start_interaction(
            instance_id=instance_id,
            ground_truth=ground_truth,
            initial_observation=initial_observation,
            **kwargs,
        )

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        inst = self._instances[instance_id]
        gt = inst["ground_truth"]
        candidates_by_idx = inst["candidates_by_idx"]

        assistant_content = ""
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "assistant":
                assistant_content = str(msg.get("content") or "")
                break

        action = self._parse_action(assistant_content)
        if action is None:
            reward = float(INVALID_ACTION_REWARD)
            obs = self._next_observation(inst, action="INVALID")
            payload = {
                "event": "invalid_action",
                "accepted_action_history": list(inst["actions"]),
                "observation": obs,
            }
            return False, json.dumps(payload, ensure_ascii=False), reward, {"invalid_action": True}

        if isinstance(action, str) and action.upper() == DEFAULT_STOP_ACTION:
            inst["actions"].append(DEFAULT_STOP_ACTION)
            solution_str = self._build_solution_str(inst["actions"])
            replay = replay_episode(ground_truth=gt, solution_str=solution_str)
            response = {
                "event": "stop",
                "accepted_action_history": list(inst["actions"]),
                "done": True,
            }
            return True, json.dumps(response, ensure_ascii=False), 0.0, {"replay": replay}

        if not isinstance(action, int) or action not in candidates_by_idx:
            reward = float(INVALID_ACTION_REWARD)
            obs = self._next_observation(inst, action="INVALID")
            payload = {
                "event": "invalid_action",
                "accepted_action_history": list(inst["actions"]),
                "observation": obs,
            }
            return False, json.dumps(payload, ensure_ascii=False), reward, {"invalid_action": True}

        if action in inst["selected_set"]:
            reward = float(INVALID_ACTION_REWARD)
            obs = self._next_observation(inst, action=action)
            payload = {
                "event": "duplicate_action",
                "accepted_action_history": list(inst["actions"]),
                "observation": obs,
            }
            return False, json.dumps(payload, ensure_ascii=False), reward, {"duplicate_action": True}

        trial_actions = list(inst["actions"]) + [action]
        replay = replay_episode(ground_truth=gt, solution_str=self._build_solution_str(trial_actions))
        step_details = replay.get("step_details") or []
        last_detail = step_details[-1] if step_details else {}
        step_reward = float((last_detail or {}).get("reward", 0.0)) if step_details else 0.0

        inst["actions"].append(action)
        inst["selected_indices"].append(action)
        inst["selected_set"].add(action)

        obs = self._next_observation(inst, action=action)
        payload = {
            "event": "accepted",
            "accepted_action_history": list(inst["actions"]),
            "observation": obs,
        }
        return False, json.dumps(payload, ensure_ascii=False), step_reward, {"replay": replay}
