"""
Memory Brain Agent — LLM agent loop for natural-language memory operations.

Accepts a natural-language request, runs tool calls in a loop via Ollama JSON mode,
and returns a synthesized answer.

The agent uses qwen3.5:9b (configurable via BRAIN_LLM_MODEL env var) on the same
Ollama instance as the extraction pipeline.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.brain.prompts import get_system_prompt
from app.brain.tools import TOOLS, TOOL_DEFINITIONS, _log_audit

logger = logging.getLogger(__name__)


def _strip_json_fences(text: str) -> str:
    """Strip markdown code fences from JSON output."""
    text = text.strip()
    if text.startswith('```'):
        first_nl = text.find('\n')
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith('```'):
            text = text.rstrip()[:-3].rstrip()
    return text


@dataclass
class BrainResult:
    """Result of a brain agent invocation."""
    answer: str
    steps: int
    tools_called: List[str] = field(default_factory=list)
    user_id: str = ""
    success: bool = True
    error: Optional[str] = None
    # For confirmation flow
    requires_confirmation: bool = False
    planned_actions: Optional[List[Dict]] = None
    elapsed_seconds: float = 0.0


class MemoryBrainAgent:
    """
    LLM agent that sits between callers and the three databases.
    Accepts natural language. Decides which tools to call. Returns synthesized answer.
    """

    DESTRUCTIVE_TOOLS = {"vector_delete", "graph_mutate", "sql_mutate"}

    def __init__(self):
        self.model = os.environ.get("BRAIN_LLM_MODEL", "qwen3.5:9b")
        self.ollama_url = os.environ.get("BRAIN_OLLAMA_URL", "http://ollama:11434")
        self.max_steps = int(os.environ.get("BRAIN_MAX_STEPS", "12"))

    def run(
        self,
        request: str,
        user_id: str,
        read_only: bool = False,
        dry_run: bool = False,
    ) -> BrainResult:
        """
        Run the brain agent loop.

        Args:
            request: Natural language request from the user/caller.
            user_id: User whose memories to operate on.
            read_only: If True, only read tools are available.
            dry_run: If True, collect destructive actions but don't execute them.
                     Returns requires_confirmation=True with planned_actions.

        Returns:
            BrainResult with the answer, steps taken, and tools called.
        """
        start = time.time()
        system_prompt = get_system_prompt(user_id, read_only=read_only)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request},
        ]

        tools_called = []
        planned_destructive = []

        for step in range(1, self.max_steps + 1):
            # Call Ollama
            try:
                response_text = self._call_ollama(messages)
            except Exception as e:
                logger.error(f"Brain step {step}: Ollama call failed: {e}")
                return BrainResult(
                    answer=f"Error: LLM call failed: {e}",
                    steps=step,
                    tools_called=tools_called,
                    user_id=user_id,
                    success=False,
                    error=str(e),
                    elapsed_seconds=time.time() - start,
                )

            # Parse JSON response
            action = self._parse_response(response_text, messages, step)
            if action is None:
                # Retry failed — return error
                return BrainResult(
                    answer="Error: Failed to parse LLM response as JSON after retries.",
                    steps=step,
                    tools_called=tools_called,
                    user_id=user_id,
                    success=False,
                    error="JSON parse failure",
                    elapsed_seconds=time.time() - start,
                )

            # Check if final
            if action.get("final"):
                answer = action.get("answer", "No answer provided.")
                _log_audit("brain_run", user_id,
                           {"request": request[:500], "steps": step, "tools": tools_called},
                           {"answer": answer[:500]})
                return BrainResult(
                    answer=answer,
                    steps=step,
                    tools_called=tools_called,
                    user_id=user_id,
                    success=True,
                    elapsed_seconds=time.time() - start,
                )

            # Execute tool
            tool_name = action.get("action")
            tool_args = action.get("args") or {}

            if not tool_name or tool_name not in TOOLS:
                # Unknown tool — tell the brain
                error_msg = f"Unknown tool: {tool_name}. Available: {', '.join(TOOLS.keys())}"
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": json.dumps({"error": error_msg})})
                continue

            tools_called.append(tool_name)

            # Dry-run mode: collect destructive actions instead of executing
            if dry_run and tool_name in self.DESTRUCTIVE_TOOLS:
                planned_destructive.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "thinking": action.get("thinking", ""),
                })
                # Tell the brain it was "executed" so it can continue planning
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": json.dumps({
                    "result": f"[DRY RUN] {tool_name} would be executed with args: {json.dumps(tool_args)}"
                })})
                continue

            # Read-only mode: block write tools
            if read_only and tool_name in self.DESTRUCTIVE_TOOLS | {"vector_store"}:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": json.dumps({
                    "error": f"Tool {tool_name} is not available in read-only mode."
                })})
                continue

            # Execute the tool
            try:
                tool_fn = TOOLS[tool_name]
                result = tool_fn(**tool_args)
                result_str = json.dumps(result, default=str)
            except Exception as e:
                logger.warning(f"Brain step {step}: tool {tool_name} failed: {e}")
                result_str = json.dumps({"error": str(e)})

            # Append to conversation
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": result_str})

        # Reached max steps
        if dry_run and planned_destructive:
            return BrainResult(
                answer="Plan collected (dry run). Confirm to execute.",
                steps=self.max_steps,
                tools_called=tools_called,
                user_id=user_id,
                success=True,
                requires_confirmation=True,
                planned_actions=planned_destructive,
                elapsed_seconds=time.time() - start,
            )

        _log_audit("brain_run", user_id,
                   {"request": request[:500], "steps": self.max_steps, "tools": tools_called},
                   {"answer": "Reached step limit"})
        return BrainResult(
            answer="Reached maximum steps without completing. Partial results may be in the tool calls above.",
            steps=self.max_steps,
            tools_called=tools_called,
            user_id=user_id,
            success=False,
            error="max_steps_reached",
            elapsed_seconds=time.time() - start,
        )

    def _call_ollama(self, messages: List[Dict]) -> str:
        """Call Ollama chat API with JSON mode."""
        import requests

        payload = {
            "model": self.model,
            "messages": messages,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 20,
                "num_predict": 1024,
            },
        }

        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()

        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return content

    def _parse_response(self, response_text: str, messages: List[Dict], step: int) -> Optional[Dict]:
        """
        Parse JSON response from the brain LLM.
        Retries up to 2 times if JSON parsing fails.
        """
        max_retries = 2
        current_text = response_text

        for attempt in range(max_retries + 1):
            try:
                cleaned = _strip_json_fences(current_text)
                parsed = json.loads(cleaned)

                # Validate minimum structure
                if not isinstance(parsed, dict):
                    raise ValueError("Response is not a JSON object")
                if "final" not in parsed and "action" not in parsed:
                    raise ValueError("Missing 'final' or 'action' field")

                return parsed

            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries:
                    logger.warning(f"Brain step {step}, attempt {attempt+1}: JSON parse failed: {e}")
                    # Ask the LLM to fix its output
                    messages.append({"role": "assistant", "content": current_text})
                    messages.append({"role": "user", "content": json.dumps({
                        "error": f"Invalid JSON: {str(e)}. You MUST respond with a single valid JSON object matching the schema: {{\"thinking\": \"...\", \"action\": \"tool_name\", \"args\": {{}}, \"final\": false}} or {{\"thinking\": \"...\", \"final\": true, \"answer\": \"...\"}}"
                    })})
                    try:
                        current_text = self._call_ollama(messages)
                    except Exception:
                        return None
                else:
                    logger.error(f"Brain step {step}: JSON parse failed after {max_retries} retries")
                    return None

        return None

    def run_confirmed(
        self,
        request: str,
        user_id: str,
        planned_actions: List[Dict],
    ) -> BrainResult:
        """
        Execute previously planned destructive actions (from a dry_run).
        Runs each planned tool call directly, then returns a summary.
        """
        start = time.time()
        tools_called = []
        results = []

        for action in planned_actions:
            tool_name = action["tool"]
            tool_args = action.get("args", {})
            tools_called.append(tool_name)

            try:
                tool_fn = TOOLS[tool_name]
                result = tool_fn(**tool_args)
                results.append({"tool": tool_name, "result": result})
            except Exception as e:
                results.append({"tool": tool_name, "error": str(e)})

        # Summarize
        summary_parts = []
        for r in results:
            if "error" in r:
                summary_parts.append(f"- {r['tool']}: ERROR: {r['error']}")
            else:
                summary_parts.append(f"- {r['tool']}: {json.dumps(r['result'], default=str)[:200]}")

        answer = f"Executed {len(planned_actions)} planned actions:\n" + "\n".join(summary_parts)

        _log_audit("brain_run_confirmed", user_id,
                   {"request": request[:500], "actions": len(planned_actions)},
                   {"answer": answer[:500]})

        return BrainResult(
            answer=answer,
            steps=len(planned_actions),
            tools_called=tools_called,
            user_id=user_id,
            success=all("error" not in r for r in results),
            elapsed_seconds=time.time() - start,
        )


# Module-level singleton
brain_agent = MemoryBrainAgent()
