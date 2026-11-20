"""Pinned GenEvolve loop with only the command serialization substituted.

The execution, image-ID management, finalization, retry/last-round behavior
and termination fields are inherited unchanged. See the paired-loop tests.
"""

from .common import load_upstream
from .action_protocol import FORMAT_ERROR, convert_instruction, parse_generate, parse_tool

load_upstream()
from genevolve.agent import GenEvolveAgent, GenEvolveResult, ImageIdManager
from genevolve.system_prompt import FINAL_STEP_MESSAGE, SYSTEM_PROMPT


class CommandAgent(GenEvolveAgent):
    def __init__(self, *args, system_prompt=None, **kwargs):
        super().__init__(*args, system_prompt=convert_instruction(system_prompt or SYSTEM_PROMPT), **kwargs)

    def run(self, user_prompt):
        # This is the pinned author's run() control flow, not a new rollout.
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        img_manager = ImageIdManager()
        rounds = 0
        error = None
        termination = "max_rounds_reached"
        gen_prompt = ""
        reference_images = []

        while rounds < self.max_rounds:
            rounds += 1
            is_last = rounds == self.max_rounds
            if is_last:
                messages.append({"role": "user", "content": convert_instruction(FINAL_STEP_MESSAGE)})
            try:
                response = self._call_llm(messages)
            except Exception as exc:
                error = f"llm_call_failed: {exc}"
                termination = "error"
                break
            messages.append({"role": "assistant", "content": response})

            answer_payload = parse_generate(response)
            if answer_payload is not None:
                gen_prompt, reference_images = self._finalize_answer(
                    answer_payload, img_manager, self.n_max_reference_images,
                )
                termination = "answer"
                break
            tool = parse_tool(response)
            if tool is None:
                if is_last:
                    error = "no_answer_in_final_round"
                    termination = "no_answer"
                    break
                messages.append({"role": "user", "content": FORMAT_ERROR})
                continue
            tool_name, tool_args = tool
            try:
                tool_obs = self._exec_tool(tool_name, tool_args, img_manager)
            except Exception as exc:
                tool_obs = f"[tool error] {tool_name}: {exc}"
            messages.append({"role": "user", "content": f"<tool_response>\n{tool_obs}\n</tool_response>"})

        return GenEvolveResult(
            prompt=user_prompt, gen_prompt=gen_prompt, reference_images=reference_images,
            messages=messages, termination=termination, rounds=rounds, error=error,
        )
