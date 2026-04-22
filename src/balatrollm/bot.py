"""Core LLM-powered Balatro bot implementation."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
from openai.types.chat import ChatCompletion

from .client import BalatroClient, BalatroError
from .collector import (
    ChatCompletionError,
    ChatCompletionResponse,
    Collector,
    FinishReason,
    Stats,
)
from .config import Config, Task, get_model_config
from .deterministic_player import DeterministicPlayer
from .llm import LLMClient, LLMClientError, LLMTimeoutError
from .strategy import StrategyManager

logger = logging.getLogger(__name__)


class BotError(Exception):
    """Base exception for bot errors."""

    pass


class Bot:
    """One-shot LLM-powered Balatro bot. Creates clients, plays a single game, returns stats."""

    def __init__(self, task: Task, config: Config, port: int | None = None) -> None:
        self.task = task
        self.config = config
        self.port = port if port is not None else config.port
        self.model_config = get_model_config(config.model_config)
        self.strategy = StrategyManager(task.strategy)

        self._balatro: BalatroClient | None = None
        self._llm: LLMClient | None = None
        self._collector: Collector | None = None

        self._last_error_msg: str | None = None
        self._last_failed_msg: str | None = None
        self._history: list[dict[str, Any]] = []

        # Finish reason tracking
        self._finish_reason: FinishReason | None = None
        # Separate counters for error calls vs failed calls
        self._consecutive_errors: int = 0
        self._consecutive_faileds: int = 0
        
        # Deterministic player state
        self._deterministic_player = DeterministicPlayer()
        self._current_game_plan: dict[str, Any] | None = None

    async def __aenter__(self) -> "Bot":
        """Initialize all clients."""
        self._balatro = BalatroClient(
            host=self.config.host,
            port=self.port,
        )
        await self._balatro.__aenter__()

        self._llm = LLMClient(
            base_url=self.config.base_url,
            api_key=self.config.api_key or "",
        )
        await self._llm.__aenter__()

        return self

    async def __aexit__(self, *_: Any) -> None:
        """Clean up all clients."""
        if self._llm is not None:
            await self._llm.__aexit__(None, None, None)
            self._llm = None
        if self._balatro is not None:
            await self._balatro.__aexit__(None, None, None)
            self._balatro = None

    def _setup_file_logging(self) -> None:
        """Redirect logging to file in run directory."""
        if self._collector is None:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        log_file = self._collector.run_dir / "run.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

    async def _wait_for_menu(self, timeout: float = 10.0) -> None:
        """Wait for game to be in MENU state."""
        assert self._balatro is not None

        start = time.time()
        while time.time() - start < timeout:
            try:
                gamestate = await self._balatro.call("gamestate")
                if gamestate.get("state", "") == "MENU":
                    logger.debug("Confirmed MENU state")
                    return
            except Exception as e:
                logger.debug(f"Gamestate check failed: {e}")
            await asyncio.sleep(0.5)

        self._finish_reason = "connection_abort"
        raise BotError(f"Timeout waiting for MENU state after {timeout}s")

    async def play(self, runs_dir: Path = Path.cwd()) -> Stats:
        """Play a single game run. Returns final Stats."""
        if self._balatro is None or self._llm is None:
            raise RuntimeError(
                "Bot not initialized. Use 'async with Bot(config) as bot:'"
            )

        # Health check before initializing collector
        try:
            await self._balatro.call("gamestate")
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            self._finish_reason = "connection_abort"
            raise BotError(
                f"Cannot connect to Balatro on {self.config.host}:{self.port}. "
                "Make sure Balatro instance started correctly."
            ) from e
        except Exception as e:
            self._finish_reason = "connection_abort"
            raise BotError(f"Failed to connect to Balatro: {e}") from e

        self._collector = Collector(self.task, runs_dir)
        self._setup_file_logging()

        logger.info("Starting game")
        logger.info(f"Run data will be saved to: {self._collector.run_dir}")

        try:
            await self._balatro.call("menu")
            await self._wait_for_menu()
            gamestate = await self._balatro.call(
                "start",
                {
                    "deck": self.task.deck,
                    "stake": self.task.stake,
                    "seed": self.task.seed,
                },
            )
            await self._run_game_loop(gamestate)
        except BotError:
            logger.error("Game ended due to bot error")
            raise
        except Exception as e:
            self._finish_reason = "unexpected_error"
            logger.exception("Unexpected error occurred during gameplay")
            raise BotError(f"Unexpected error: {e}") from e
        finally:
            if self._collector:
                try:
                    reason: FinishReason = self._finish_reason or "unexpected_error"
                    self._collector.write_stats(reason)
                    logger.info("Stats written")
                except Exception as e:
                    logger.debug(
                        f"Could not write stats (normal if run failed early): {e}"
                    )

        return self._collector._calculate_stats(
            self._finish_reason or "unexpected_error"
        )

    async def _run_game_loop(self, gamestate: dict[str, Any]) -> None:
        """Main game loop."""
        assert self._balatro is not None
        assert self._llm is not None
        assert self._collector is not None

        while True:
            if gamestate.get("won", False):
                self._finish_reason = "won"
                logger.info("Game won! Waiting for GAME_OVER state...")
                break

            current_state = gamestate.get("state", "")
            logger.info(f"State: {current_state}")

            await asyncio.sleep(0.5)
            await self._balatro.call("gamestate")

            match current_state:
                case "SELECTING_HAND":
                    if self.config.strategy != "original":
                        if not self._current_game_plan:
                            logger.info("No game plan found for round. Asking LLM for game plan...")
                            plan_gamestate = gamestate.copy()
                            plan_gamestate["state"] = "GAME_PLANNING"
                            response = await self._get_llm_response(plan_gamestate)
                            llm_payouts = self._parse_game_plan(response)
                            if llm_payouts:
                                logger.info(f"LLM provided Payouts: {llm_payouts['target_hands']}")
                                logger.info("Running Genetic Algorithm Strategy Optimizer...")
                                optimized_plan = self._deterministic_player.optimize_strategy(gamestate, llm_payouts["target_hands"])
                                self._current_game_plan = optimized_plan
                                self._collector.record_call("successful")
                            else:
                                logger.warning("LLM failed to provide payouts. Using fallback.")
                                fallback_payouts = {
                                    "Straight Flush": 5000, "Four of a Kind": 3000, "Full House": 1500, 
                                    "Flush": 1200, "Straight": 1000, "Three of a Kind": 500, 
                                    "Two Pair": 300, "Pair": 100, "High Card": 50
                                }
                                logger.info("Running Genetic Algorithm Strategy Optimizer with Fallback Payouts...")
                                optimized_plan = self._deterministic_player.optimize_strategy(gamestate, fallback_payouts)
                                self._current_game_plan = optimized_plan
                                self._collector.record_call("failed")
                            self._deterministic_player.reset_round()
                            logger.info(f"Optimized Game plan set: {self._current_game_plan}")

                        action = self._deterministic_player.decide(gamestate, self._current_game_plan)
                        logger.info(f"Deterministic action: {action['method']} {action['params']}")
                        gamestate = await self._balatro.call(action["method"], action["params"])
                        self._collector.write_gamestate(gamestate)
                        self._history.append({"method": action["method"], "params": action["params"]})
                        continue
                        
                    # If strategy == "original", fall through to the LLM response handler
                    response = await self._get_llm_response(gamestate)
                    gamestate = await self._execute_tool_call(response)
                    
                case "SHOP" | "SMODS_BOOSTER_OPENED":
                    if current_state == "SHOP":
                        money = gamestate.get("money", 0)
                        min_cost = gamestate.get("round", {}).get("reroll_cost", 5)
                        
                        for list_key in ["shop", "vouchers", "packs"]:
                            for item in gamestate.get(list_key, {}).get("cards", []):
                                if isinstance(item, dict):
                                    state = item.get("state", {})
                                    is_hidden = isinstance(state, dict) and state.get("hidden", False)
                                    if not is_hidden:
                                        cost = item.get("cost", {})
                                        buy_cost = cost.get("buy", 999) if isinstance(cost, dict) else 999
                                        min_cost = min(min_cost, buy_cost)
                        
                        if money < min_cost:
                            logger.info(f"Not enough money (${money} < min cost ${min_cost}). Auto-skipping shop.")
                            gamestate = await self._balatro.call("next_round", {})
                            self._collector.write_gamestate(gamestate)
                            self._history.append({"method": "next_round", "params": {}})
                            continue

                    response = await self._get_llm_response(gamestate)
                    gamestate = await self._execute_tool_call(response)
                case "ROUND_EVAL":
                    gamestate = await self._balatro.call("cash_out")
                case "BLIND_SELECT":
                    self._current_game_plan = None
                    # NOTE: This bot always selects and never skips blinds
                    gamestate = await self._balatro.call("select")
                case "GAME_OVER":
                    self._finish_reason = "lost"
                    logger.info("Game over!")
                    break
                case _:
                    await asyncio.sleep(1)
                    gamestate = await self._balatro.call("gamestate")

    async def _get_llm_response(self, gamestate: dict[str, Any]) -> ChatCompletion:
        """Get LLM response for current game state."""
        assert self._balatro is not None
        assert self._llm is not None
        assert self._collector is not None

        strategy_content = self.strategy.render_strategy(gamestate)
        gamestate_content = self.strategy.render_gamestate(gamestate)
        memory_content = self.strategy.render_memory(
            history=self._history[-10:],
            last_error=self._last_error_msg,
            last_failure=self._last_failed_msg,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": strategy_content,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": gamestate_content},
                    {"type": "text", "text": memory_content},
                ],
            }
        ]

        tools = self.strategy.get_tools(gamestate["state"])

        request_data = {
            "model": self.task.model,
            "messages": messages,
            "tools": tools,
            **self.model_config,
        }

        # Force the model to use set_game_plan during GAME_PLANNING
        if gamestate["state"] == "GAME_PLANNING" and tools:
            request_data["tool_choice"] = {"type": "function", "function": {"name": "set_game_plan"}}

        custom_id = self._collector.write_request(request_data)
        request_id = str(time.time_ns() // 1_000_000)

        try:
            response = await self._llm.call(
                model=self.task.model,
                messages=messages,
                tools=tools,
                model_config=self.model_config,
            )

            try:
                await self._balatro.call(
                    "screenshot",
                    {"path": str(self._collector.screenshot_dir / f"{custom_id}.png")},
                )
            except BalatroError as e:
                logger.warning(f"Screenshot failed: {e}")

            self._collector.write_response(
                id=str(time.time_ns() // 1_000_000),
                custom_id=custom_id,
                response=ChatCompletionResponse(
                    request_id=request_id,
                    status_code=200,
                    body=response.model_dump(),
                ),
            )

            return response

        except LLMTimeoutError as e:
            self._collector.write_response(
                id=str(time.time_ns() // 1_000_000),
                custom_id=custom_id,
                error=ChatCompletionError(code="timeout", message=str(e)),
            )
            self._finish_reason = "llm_abort"
            raise BotError("3 consecutive LLM timeouts") from e

        except LLMClientError as e:
            self._collector.write_response(
                id=str(time.time_ns() // 1_000_000),
                custom_id=custom_id,
                error=ChatCompletionError(code="error", message=str(e)),
            )
            self._finish_reason = "llm_abort"
            raise BotError(f"LLM error: {e}") from e

    async def _execute_tool_call(self, response: ChatCompletion) -> dict[str, Any]:
        """Execute tool call from LLM response."""
        assert self._balatro is not None
        assert self._collector is not None

        ################################################################################
        # Parse tool call from LLM response
        ################################################################################

        message = response.choices[0].message

        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return await self._handle_error_call(
                f"No tool calls in LLM response: {message.content}"
            )

        tool_call = message.tool_calls[0]
        function_obj = getattr(tool_call, "function", tool_call)

        fn_name = getattr(function_obj, "name", None)
        if not fn_name:
            return await self._handle_error_call("Invalid tool call: missing name")

        fn_args_str = getattr(function_obj, "arguments", None)
        if not fn_args_str:
            return await self._handle_error_call("Invalid tool call: missing arguments")

        try:
            fn_args = json.loads(fn_args_str)
        except json.JSONDecodeError as e:
            return await self._handle_error_call(
                f"Invalid JSON in tool call arguments: {e}"
            )

        ################################################################################
        # Execute tool call
        ################################################################################

        try:
            logger.info(f"Executing: {fn_name}({fn_args})")
            gamestate = await self._balatro.call(fn_name, fn_args)

            self._collector.reset_failures()
            # Reset both consecutive counters on success
            self._consecutive_errors = 0
            self._consecutive_faileds = 0
            self._last_error_msg = None
            self._last_failed_msg = None
            self._collector.record_call("successful")
            self._collector.write_gamestate(gamestate)
            self._history.append({"method": fn_name, "params": fn_args})

            return gamestate

        except BalatroError as e:
            return await self._handle_failed_call(f"BalatroError: {e}")

        except httpx.TransportError as e:
            logger.warning(f"Game transport error during tool call: {e}")
            self._collector.record_call("failed")
            try:
                return await self._balatro.call("gamestate")
            except Exception:
                self._finish_reason = "connection_abort"
                raise BotError(f"Game unresponsive after transport error: {e}") from e

    async def _handle_error_call(self, msg: str) -> dict[str, Any]:
        """Handle invalid LLM response (no valid tool call)."""
        assert self._balatro is not None
        assert self._collector is not None

        logger.warning(f"Error call: {msg}")
        self._last_error_msg = msg
        self._collector.record_failure()
        self._collector.record_call("error")

        # Track consecutive error calls separately
        self._consecutive_errors += 1
        self._consecutive_faileds = 0

        if self._consecutive_errors >= Collector.MAX_CONSECUTIVE_FAILURES:
            self._finish_reason = "consecutive_error_calls"
            raise BotError("Too many consecutive error calls")

        return await self._balatro.call("gamestate")

    async def _handle_failed_call(self, msg: str) -> dict[str, Any]:
        """Handle valid tool call that resulted in BalatroError."""
        assert self._balatro is not None
        assert self._collector is not None

        logger.warning(f"Failed call: {msg}")
        self._last_failed_msg = msg
        self._collector.record_failure()
        self._collector.record_call("failed")

        # Track consecutive failed calls separately
        self._consecutive_faileds += 1
        self._consecutive_errors = 0

        if self._consecutive_faileds >= Collector.MAX_CONSECUTIVE_FAILURES:
            self._finish_reason = "consecutive_failed_calls"
            raise BotError("Too many consecutive failed calls")

        return await self._balatro.call("gamestate")

    def _parse_game_plan(self, response: ChatCompletion) -> dict[str, Any] | None:
        """Parse game plan tool call from LLM response."""
        try:
            message = response.choices[0].message
            
            # Check for valid tool call
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_call = message.tool_calls[0]
                function_obj = getattr(tool_call, "function", tool_call)
                
                if getattr(function_obj, "name", None) == "set_game_plan":
                    args_str = getattr(function_obj, "arguments", "")
                    args = json.loads(args_str)
                    return {
                        "target_hands": args.get("target_hands", {}),
                        "discard_aggressiveness": args.get("discard_aggressiveness", 0.5)
                    }
            
            # Fallback: Try to parse JSON from content
            content = message.content or getattr(message, "reasoning_content", "")
            if content:
                import re
                json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
                if json_match:
                    args = json.loads(json_match.group(1))
                    return {
                        "target_hands": args.get("target_hands", {}),
                        "discard_aggressiveness": args.get("discard_aggressiveness", 0.5)
                    }
                
                json_match = re.search(r"\{[\s\S]*\"target_hands\"[\s\S]*\}", content)
                if json_match:
                    args = json.loads(json_match.group(0))
                    return {
                        "target_hands": args.get("target_hands", {}),
                        "discard_aggressiveness": args.get("discard_aggressiveness", 0.5)
                    }
            
            return None
        except Exception as e:
            logger.warning(f"Failed to parse game plan: {e}")
            return None
