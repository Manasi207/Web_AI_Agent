"""
web_multi_agent_mcts.py

A hierarchical multi-agent web agent using Monte-Carlo Tree Search (MCTS) + LLM guidance.
- Global MCTS decides agent workflow (which agents to run / sequence).
- Per-agent MCTS decides micro-actions inside each agent (DOM clicks, input, parse methods).
- Optional OpenAI LLM integration (via OPENAI_API_KEY) to suggest or score actions.
- Uses Playwright synchronous API; designed to run in same venv where playwright is available.

Usage:
    python web_multi_agent_mcts.py

Configuration via environment variables:
    OPENAI_API_KEY (optional) - if set, the script will call OpenAI for suggestions/scoring.
"""
import os
import time
import math
import random
from typing import List, Optional, Any, Dict
from playwright.sync_api import sync_playwright

# optional import for LLM; fallback if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------------------------
# Utilities and LLM Connector
# ---------------------------
class LLMConnector:
    """
    Simple LLM connector:
    - If OPENAI_API_KEY is set and openai package present, uses OpenAI completions (text-davinci-003-like).
    - Otherwise falls back to heuristic functions.
    """
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key and OPENAI_AVAILABLE:
            openai.api_key = self.api_key
            self.mode = "openai"
        else:
            self.mode = "heuristic"

    def score_action(self, context: str, action_text: str) -> float:
        """
        Returns a score (0..1) representing how promising an action is given context.
        If LLM available, asks for a short judgement; else uses heuristics (e.g., prefer 'search', 'next', 'submit', 'book', etc).
        """
        if self.mode == "openai":
            prompt = (
                f"You are a short-scoring assistant. Given the page context and a candidate action, "
                f"return a single number from 0.0 to 1.0 indicating how likely this action helps achieve the goal.\n\n"
                f"Context:\n{context[:1000]}\n\nAction: {action_text}\n\nReturn only a number between 0.0 and 1.0."
            )
            try:
                resp = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=8,
                    temperature=0.0,
                )
                text = resp.choices[0].text.strip()
                # parse float robustly
                for token in text.split():
                    try:
                        val = float(token)
                        return max(0.0, min(1.0, val))
                    except:
                        continue
            except Exception:
                pass
            return 0.5
        else:
            # heuristic fallback
            lower = action_text.lower()
            score = 0.5
            boost_keywords = ["search", "next", "submit", "go", "book", "confirm", "continue", "login", "find", "search"]
            penalty_keywords = ["cancel", "close", "dismiss", "back"]
            for kw in boost_keywords:
                if kw in lower:
                    score += 0.2
            for kw in penalty_keywords:
                if kw in lower:
                    score -= 0.25
            # clamp
            return max(0.0, min(1.0, score))

llm = LLMConnector()


# ---------------------------
# MCTS core classes (generic)
# ---------------------------
class ActionNode:
    def __init__(self, state: 'WebState', parent: Optional['ActionNode']=None, action: Optional[str]=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[ActionNode] = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self) -> bool:
        avail = self.state.get_available_actions()
        return len(self.children) >= len(avail)

    def best_child(self, exploration_weight=1.41) -> 'ActionNode':
        # UCT with small epsilon to avoid div-by-zero
        best = max(
            self.children,
            key=lambda c: (c.value / (c.visits + 1e-9)) +
                          exploration_weight * math.sqrt(2 * math.log(self.visits + 1) / (c.visits + 1e-9))
        )
        return best

# ---------------------------
# WebState abstraction
# ---------------------------
class WebState:
    """
    Represents a lightweight snapshot of page for dry-run simulations.
    For live actions, we keep a 'page' reference and live=True.
    For dry simulations, page=None and we operate on 'snapshot' (list of available actions).
    """
    def __init__(self, page=None, url: str = "", snapshot: Optional[Dict[str, Any]] = None, live: bool = True):
        self.page = page            # Playwright page object (only for live states)
        self.url = url
        self.live = live
        # snapshot contains a list of action_texts and optionally content string
        self.snapshot = snapshot if snapshot is not None else {"actions": [], "content": ""}

    def refresh_snapshot_from_live(self):
        """Build a simple snapshot (available actions & content) from the live Playwright page."""
        if not self.page:
            return
        try:
            buttons = self.page.query_selector_all("button, a, input[type=submit]")
            actions = []
            for b in buttons:
                try:
                    if b.is_visible():
                        text = b.inner_text().strip()
                        if not text:
                            # fallback: try value attribute for inputs
                            val = b.get_attribute("value") or ""
                            text = val.strip()
                        if text:
                            actions.append(text)
                except Exception:
                    continue
            content = self.page.content()[:4000]  # keep bounded
            self.snapshot = {"actions": actions, "content": content}
        except Exception as e:
            # if something fails, keep old snapshot
            print(f"[snapshot] refresh failed: {e}")

    def get_available_actions(self) -> List[str]:
        return list(self.snapshot.get("actions", []))

    def apply_action_live(self, action_text: str, timeout_ms: int = 1500):
        """Perform a real click on the live page. Block until after click + short wait."""
        if not self.page:
            raise RuntimeError("No live page available for live action.")
        try:
            # find clickable element by text
            el = self.page.query_selector(f'button:has-text("{action_text}")') or \
                 self.page.query_selector(f'a:has-text("{action_text}")') or \
                 self.page.query_selector(f'input[value="{action_text}"]')
            if el:
                el.click()
                # small wait for navigation or dynamic changes
                self.page.wait_for_timeout(timeout_ms)
                self.refresh_snapshot_from_live()
                print(f"[LIVE] Clicked: {action_text}")
                return True
            else:
                print(f"[LIVE] No live element matched: {action_text}")
                return False
        except Exception as e:
            print(f"[LIVE] Error clicking {action_text}: {e}")
            return False

    def simulate_apply_action(self, action_text: str):
        """
        For simulation: we do not mutate real page. We approximate state transitions by removing the action from snapshot.
        We also allow LLM to score action and adjust reward heuristics.
        """
        # naive simulation: remove action from list and append a "result" pseudo-action
        actions = list(self.snapshot.get("actions", []))
        if action_text in actions:
            actions.remove(action_text)
        # optionally add some synthetic new actions to simulate page change (small branching)
        synthetic_new = []
        if random.random() < 0.3:
            synthetic_new.append(action_text + " → next")
        self.snapshot["actions"] = actions + synthetic_new
        # keep content unchanged in simulation; LLM can be used for scoring separately
        return

    def clone(self) -> 'WebState':
        # For simulation, produce a copy with same snapshot and no live page
        copied_snapshot = {"actions": list(self.snapshot.get("actions", [])),
                           "content": str(self.snapshot.get("content", ""))}
        return WebState(page=None, url=self.url, snapshot=copied_snapshot, live=False)

    def is_goal_state(self, goal_text: str) -> bool:
        return goal_text.lower() in self.snapshot.get("content", "").lower()


# ---------------------------
# Agent abstraction & specific agents
# ---------------------------
class BaseAgent:
    """
    Base agent class. Each agent implements:
    - suggest_local_actions(state): list of candidate action texts
    - evaluate_goal(state): float [0..1]
    - act_live(state, action_text): perform a real action on page when executing chosen plan
    """
    def __init__(self, name: str, llm: LLMConnector, rollout_depth: int = 3, iterations: int = 20):
        self.name = name
        self.llm = llm
        self.rollout_depth = rollout_depth
        self.iterations = iterations

    def suggest_local_actions(self, state: WebState) -> List[str]:
        """Default: return available actions from the state snapshot."""
        return state.get_available_actions()

    def evaluate_goal(self, state: WebState, goal_text: str) -> float:
        """Return a heuristic score of how close this state is to goal."""
        # default: check if goal text in content
        return 1.0 if state.is_goal_state(goal_text) else 0.0

    def act_live(self, state: WebState, action_text: str) -> bool:
        """Perform the action on the live page."""
        return state.apply_action_live(action_text)


class NavigatorAgent(BaseAgent):
    def __init__(self, llm, rollout_depth=3, iterations=25):
        super().__init__("Navigator", llm, rollout_depth, iterations)

    def suggest_local_actions(self, state: WebState) -> List[str]:
        # add a preference for navigation-type actions
        actions = state.get_available_actions()
        # score via LLM and sort
        scored = [(a, self.llm.score_action(state.snapshot.get("content", ""), a)) for a in actions]
        scored_sorted = [a for a, s in sorted(scored, key=lambda x: -x[1])]
        return scored_sorted


class ExtractorAgent(BaseAgent):
    def __init__(self, llm, rollout_depth=2, iterations=15):
        super().__init__("Extractor", llm, rollout_depth, iterations)

    def suggest_local_actions(self, state: WebState) -> List[str]:
        # extractor cares about "view details", "more", "expand" etc.
        actions = state.get_available_actions()
        priority = []
        for a in actions:
            if any(k in a.lower() for k in ["details", "view", "more", "info", "price", "select"]):
                priority.append(a)
        # if none found, return all
        return priority + [a for a in actions if a not in priority]


class ValidatorAgent(BaseAgent):
    def __init__(self, llm, rollout_depth=2, iterations=10):
        super().__init__("Validator", llm, rollout_depth, iterations)

    def evaluate_goal(self, state: WebState, goal_text: str) -> float:
        # validator looks for explicit confirmations, messages, or goal_text in content
        content = state.snapshot.get("content", "").lower()
        if goal_text.lower() in content:
            return 1.0
        if any(x in content for x in ["success", "confirmed", "booked", "thank you", "order received"]):
            return 0.9
        # else use LLM scoring as fallback
        return self.llm.score_action(content, "is this a confirmation?")

# ---------------------------
# Per-agent local MCTS
# ---------------------------
class LocalMCTS:
    def __init__(self, agent: BaseAgent, goal_text: str, exploration_weight=1.0):
        self.agent = agent
        self.goal_text = goal_text
        self.exploration_weight = exploration_weight

    def search(self, root_state: WebState) -> Optional[str]:
        root_node = ActionNode(root_state)
        for _ in range(self.agent.iterations):
            node = self._select(root_node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
        # select best child (no exploration)
        if not root_node.children:
            return None
        best = root_node.best_child(exploration_weight=0)
        return best.action

    def _select(self, node: ActionNode) -> ActionNode:
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.exploration_weight)
        return self._expand(node)

    def _expand(self, node: ActionNode) -> ActionNode:
        actions = self.agent.suggest_local_actions(node.state)
        for act in actions:
            if all(child.action != act for child in node.children):
                # create a simulation state copy (dry)
                new_state = node.state.clone()
                # apply simulated action (doesn't touch real page)
                new_state.simulate_apply_action(act)
                child = ActionNode(new_state, parent=node, action=act)
                node.children.append(child)
                return child
        return node

    def _simulate(self, node: ActionNode) -> float:
        state = node.state.clone()
        # rollout up to depth
        for _ in range(self.agent.rollout_depth):
            actions = self.agent.suggest_local_actions(state)
            if not actions:
                break
            act = random.choice(actions)
            state.simulate_apply_action(act)
            # early check
            if state.is_goal_state(self.goal_text):
                return 1.0
        # if reached here, use agent's heuristic evaluation
        return self.agent.evaluate_goal(state, self.goal_text)

    def _backpropagate(self, node: ActionNode, reward: float):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

# ---------------------------
# Global orchestrator MCTS (chooses agent sequence)
# ---------------------------
class OrchestratorMCTS:
    """
    Global planner that decides which agent to run next and how many times,
    using MCTS at a higher level where actions = "run Navigator", "run Extractor", "run Validator".
    """
    def __init__(self, agents: List[BaseAgent], goal_text: str, iterations: int = 30, rollout_len: int = 3):
        self.agents = agents
        self.goal_text = goal_text
        self.iterations = iterations
        self.rollout_len = rollout_len  # number of agent-steps to simulate per rollout

    def plan(self, live_state: WebState) -> List[BaseAgent]:
        """
        Returns an ordered list of agents to run in live execution.
        Planning uses dry simulations (cloned states) to evaluate sequences.
        """
        root = ActionNode(live_state.clone(), parent=None, action=None)
        # available global actions are agent names
        for _ in range(self.iterations):
            node = self._select(root)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
        # choose a short plan: select best child sequence
        plan = []
        node = root
        # pick up to rollout_len choices
        for _ in range(self.rollout_len):
            if not node.children:
                break
            child = node.best_child(exploration_weight=0)
            if child.action is None:
                break
            # find agent by name
            agent = next((a for a in self.agents if a.name == child.action), None)
            if agent:
                plan.append(agent)
            node = child
        return plan

    def _select(self, node: ActionNode) -> ActionNode:
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return self._expand(node)

    def _expand(self, node: ActionNode) -> ActionNode:
        # try adding each agent as an action
        for agent in self.agents:
            if all(child.action != agent.name for child in node.children):
                # create new state clone
                new_state = node.state.clone()
                # simulate agent acting once (cheap) via applying one random suggested action
                # pick candidate actions and simulate one
                candidate_actions = agent.suggest_local_actions(new_state)
                if candidate_actions:
                    chosen = random.choice(candidate_actions)
                    new_state.simulate_apply_action(chosen)
                child = ActionNode(new_state, parent=node, action=agent.name)
                node.children.append(child)
                return child
        return node

    def _simulate(self, node: ActionNode) -> float:
        # rollout a random sequence of agents for limited length
        state = node.state.clone()
        total_reward = 0.0
        for _ in range(self.rollout_len):
            agent = random.choice(self.agents)
            # agent runs small local MCTS in dry mode
            local = LocalMCTS(agent, self.goal_text)
            action_choice = local.search(state)
            if action_choice:
                state.simulate_apply_action(action_choice)
            # evaluate progress
            total_reward = max(total_reward, agent.evaluate_goal(state, self.goal_text))
            if state.is_goal_state(self.goal_text):
                return 1.0
        return total_reward

    def _backpropagate(self, node: ActionNode, reward: float):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

# ---------------------------
# Runner that integrates everything and executes chosen plan live
# ---------------------------
def run_web_multi_agent(start_url: str, goal_text: str, headless: bool = False):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        page.goto(start_url, timeout=30000)
        page.wait_for_load_state("load")
        print(f"[🌐] Loaded: {start_url}")

        live_state = WebState(page=page, url=start_url, snapshot={"actions": [], "content": ""}, live=True)
        live_state.refresh_snapshot_from_live()

        # create agents
        navigator = NavigatorAgent(llm, rollout_depth=3, iterations=20)
        extractor = ExtractorAgent(llm, rollout_depth=2, iterations=12)
        validator = ValidatorAgent(llm, rollout_depth=2, iterations=10)
        agents = [navigator, extractor, validator]

        orchestrator = OrchestratorMCTS(agents, goal_text, iterations=30, rollout_len=3)

        # plan (global)
        plan = orchestrator.plan(live_state)
        print("[ORCH] Planned agent order:", [a.name for a in plan])

        # execute planned agents live, with local MCTS per agent for micro-decisions
        for agent in plan:
            print(f"[EXEC] Running agent: {agent.name}")
            # refresh live snapshot before each agent run
            live_state.refresh_snapshot_from_live()
            local_mcts = LocalMCTS(agent, goal_text)
            chosen_action = local_mcts.search(live_state)
            if chosen_action:
                print(f"[{agent.name}] Chosen action (live): {chosen_action}")
                success = agent.act_live(live_state, chosen_action)
                # after live action, re-evaluate goal
                live_state.refresh_snapshot_from_live()
                score = agent.evaluate_goal(live_state, goal_text)
                print(f"[{agent.name}] Post-action score for goal '{goal_text}': {score:.2f}")
                if score >= 0.95 or live_state.is_goal_state(goal_text):
                    print(f"[SUCCESS] Goal appears achieved after agent {agent.name}.")
                    break
            else:
                print(f"[{agent.name}] No chosen local action (skipping).")

        print("[RUN] Final page content snippet:")
        print(live_state.snapshot.get("content", "")[:1000])
        browser.close()

# ---------------------------
# If run as script
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a multi-agent MCTS web agent.")
    parser.add_argument("--url", type=str, default="https://www.google.com", help="Start URL")
    parser.add_argument("--goal", type=str, default="News", help="Goal text to find on page")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    args = parser.parse_args()

    run_web_multi_agent(args.url, args.goal, headless=args.headless)
