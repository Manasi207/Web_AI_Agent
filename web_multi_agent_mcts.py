"""
web_multi_agent_mcts.py

A hierarchical multi-agent web agent using Monte-Carlo Tree Search (MCTS) + LLM guidance
or a simple crawler mode.

Modes:
- Goal-driven: Global MCTS decides agent workflow (Navigator/Extractor/Validator).
- Crawl: Visits all links of a given site, stays on the last page.

Usage:
    python web_multi_agent_mcts.py --url "https://example.com" --goal "News"
    python web_multi_agent_mcts.py --url "https://example.com" --crawl
"""

import os
import time
import math
import random
from typing import List, Optional, Any, Dict
from urllib.parse import urlparse, urljoin
from playwright.sync_api import sync_playwright

# optional import for LLM
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------------------------
# Utilities and LLM Connector
# ---------------------------
class LLMConnector:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key and OPENAI_AVAILABLE:
            openai.api_key = self.api_key
            self.mode = "openai"
        else:
            self.mode = "heuristic"

    def score_action(self, context: str, action_text: str) -> float:
        if self.mode == "openai":
            prompt = (
                f"You are a short-scoring assistant. Given the page context and a candidate action, "
                f"return a single number from 0.0 to 1.0 indicating how likely this action helps achieve the goal.\n\n"
                f"Context:\n{context[:1000]}\n\nAction: {action_text}\n\nReturn only a number."
            )
            try:
                resp = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=8,
                    temperature=0.0,
                )
                text = resp.choices[0].text.strip()
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
            lower = action_text.lower()
            score = 0.5
            boost_keywords = ["search", "next", "submit", "go", "book", "confirm", "continue", "login", "find"]
            penalty_keywords = ["cancel", "close", "dismiss", "back"]
            for kw in boost_keywords:
                if kw in lower:
                    score += 0.2
            for kw in penalty_keywords:
                if kw in lower:
                    score -= 0.25
            return max(0.0, min(1.0, score))

llm = LLMConnector()

# ---------------------------
# MCTS core classes
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
        return max(
            self.children,
            key=lambda c: (c.value / (c.visits + 1e-9)) +
                          exploration_weight * math.sqrt(2 * math.log(self.visits + 1) / (c.visits + 1e-9))
        )

# ---------------------------
# WebState abstraction
# ---------------------------
class WebState:
    def __init__(self, page=None, url: str = "", snapshot: Optional[Dict[str, Any]] = None, live: bool = True):
        self.page = page
        self.url = url
        self.live = live
        self.snapshot = snapshot if snapshot is not None else {"actions": [], "content": ""}

    def refresh_snapshot_from_live(self):
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
                            val = b.get_attribute("value") or ""
                            text = val.strip()
                        if text:
                            actions.append(text)
                except Exception:
                    continue
            content = self.page.content()[:4000]
            self.snapshot = {"actions": actions, "content": content}
        except Exception as e:
            print(f"[snapshot] refresh failed: {e}")

    def get_available_actions(self) -> List[str]:
        return list(self.snapshot.get("actions", []))

    def apply_action_live(self, action_text: str, timeout_ms: int = 1500):
        if not self.page:
            raise RuntimeError("No live page available for live action.")
        try:
            el = self.page.query_selector(f'button:has-text("{action_text}")') or \
                 self.page.query_selector(f'a:has-text("{action_text}")') or \
                 self.page.query_selector(f'input[value="{action_text}"]')
            if el:
                el.click()
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
        actions = list(self.snapshot.get("actions", []))
        if action_text in actions:
            actions.remove(action_text)
        synthetic_new = []
        if random.random() < 0.3:
            synthetic_new.append(action_text + " → next")
        self.snapshot["actions"] = actions + synthetic_new
        return

    def clone(self) -> 'WebState':
        copied_snapshot = {"actions": list(self.snapshot.get("actions", [])),
                           "content": str(self.snapshot.get("content", ""))}
        return WebState(page=None, url=self.url, snapshot=copied_snapshot, live=False)

    def is_goal_state(self, goal_text: str) -> bool:
        return goal_text.lower() in self.snapshot.get("content", "").lower()

# ---------------------------
# Agents
# ---------------------------
class BaseAgent:
    def __init__(self, name: str, llm: LLMConnector, rollout_depth: int = 3, iterations: int = 20):
        self.name = name
        self.llm = llm
        self.rollout_depth = rollout_depth
        self.iterations = iterations

    def suggest_local_actions(self, state: WebState) -> List[str]:
        return state.get_available_actions()

    def evaluate_goal(self, state: WebState, goal_text: str) -> float:
        return 1.0 if state.is_goal_state(goal_text) else 0.0

    def act_live(self, state: WebState, action_text: str) -> bool:
        return state.apply_action_live(action_text)

class NavigatorAgent(BaseAgent):
    def __init__(self, llm, rollout_depth=3, iterations=25):
        super().__init__("Navigator", llm, rollout_depth, iterations)

    def suggest_local_actions(self, state: WebState) -> List[str]:
        actions = state.get_available_actions()
        scored = [(a, self.llm.score_action(state.snapshot.get("content", ""), a)) for a in actions]
        scored_sorted = [a for a, s in sorted(scored, key=lambda x: -x[1])]
        return scored_sorted

class ExtractorAgent(BaseAgent):
    def __init__(self, llm, rollout_depth=2, iterations=15):
        super().__init__("Extractor", llm, rollout_depth, iterations)

    def suggest_local_actions(self, state: WebState) -> List[str]:
        actions = state.get_available_actions()
        priority = []
        for a in actions:
            if any(k in a.lower() for k in ["details", "view", "more", "info", "price", "select"]):
                priority.append(a)
        return priority + [a for a in actions if a not in priority]

class ValidatorAgent(BaseAgent):
    def __init__(self, llm, rollout_depth=2, iterations=10):
        super().__init__("Validator", llm, rollout_depth, iterations)

    def evaluate_goal(self, state: WebState, goal_text: str) -> float:
        content = state.snapshot.get("content", "").lower()
        if goal_text.lower() in content:
            return 1.0
        if any(x in content for x in ["success", "confirmed", "booked", "thank you", "order received"]):
            return 0.9
        return self.llm.score_action(content, "is this a confirmation?")

# ---------------------------
# Local MCTS
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
                new_state = node.state.clone()
                new_state.simulate_apply_action(act)
                child = ActionNode(new_state, parent=node, action=act)
                node.children.append(child)
                return child
        return node

    def _simulate(self, node: ActionNode) -> float:
        state = node.state.clone()
        for _ in range(self.agent.rollout_depth):
            actions = self.agent.suggest_local_actions(state)
            if not actions:
                break
            act = random.choice(actions)
            state.simulate_apply_action(act)
            if state.is_goal_state(self.goal_text):
                return 1.0
        return self.agent.evaluate_goal(state, self.goal_text)

    def _backpropagate(self, node: ActionNode, reward: float):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

# ---------------------------
# Global Orchestrator MCTS
# ---------------------------
class OrchestratorMCTS:
    def __init__(self, agents: List[BaseAgent], goal_text: str, iterations: int = 60, rollout_len: int = 6):
        self.agents = agents
        self.goal_text = goal_text
        self.iterations = iterations
        self.rollout_len = rollout_len

    def plan(self, live_state: WebState) -> List[BaseAgent]:
        root = ActionNode(live_state.clone(), parent=None, action=None)
        for _ in range(self.iterations):
            node = self._select(root)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
        plan = []
        node = root
        for _ in range(self.rollout_len):
            if not node.children:
                break
            child = node.best_child(exploration_weight=0)
            if child.action is None:
                break
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
        for agent in self.agents:
            if all(child.action != agent.name for child in node.children):
                new_state = node.state.clone()
                candidate_actions = agent.suggest_local_actions(new_state)
                if candidate_actions:
                    chosen = random.choice(candidate_actions)
                    new_state.simulate_apply_action(chosen)
                child = ActionNode(new_state, parent=node, action=agent.name)
                node.children.append(child)
                return child
        return node

    def _simulate(self, node: ActionNode) -> float:
        state = node.state.clone()
        total_reward = 0.0
        for _ in range(self.rollout_len):
            agent = random.choice(self.agents)
            local = LocalMCTS(agent, self.goal_text)
            action_choice = local.search(state)
            if action_choice:
                state.simulate_apply_action(action_choice)
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
# Runner - Multi-agent
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

        navigator = NavigatorAgent(llm, rollout_depth=3, iterations=20)
        extractor = ExtractorAgent(llm, rollout_depth=2, iterations=12)
        validator = ValidatorAgent(llm, rollout_depth=2, iterations=10)
        agents = [navigator, extractor, validator]

        orchestrator = OrchestratorMCTS(agents, goal_text, iterations=60, rollout_len=6)

        plan = orchestrator.plan(live_state)
        print("[ORCH] Planned agent order:", [a.name for a in plan])

        for agent in plan:
            print(f"[EXEC] Running agent: {agent.name}")
            live_state.refresh_snapshot_from_live()
            local_mcts = LocalMCTS(agent, goal_text)
            chosen_action = local_mcts.search(live_state)
            if chosen_action:
                print(f"[{agent.name}] Chosen action (live): {chosen_action}")
                agent.act_live(live_state, chosen_action)
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

        input("\n✅ Agent finished. Press ENTER to close browser...")
        browser.close()

# ---------------------------
# Runner - Crawl all pages
# ---------------------------
def run_web_crawler(start_url: str, headless: bool = False):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        domain = urlparse(start_url).netloc

        visited = set()
        to_visit = [start_url]
        last_url = start_url

        while to_visit:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)
            last_url = url

            try:
                page.goto(url, timeout=30000)
                page.wait_for_load_state("load")
                print(f"[🌐] Visited: {url}")
            except Exception as e:
                print(f"[ERROR] Could not open {url}: {e}")
                continue

            anchors = page.query_selector_all("a")
            for a in anchors:
                try:
                    href = a.get_attribute("href")
                    if href:
                        full_url = urljoin(url, href)
                        if urlparse(full_url).netloc == domain:
                            if full_url not in visited:
                                to_visit.append(full_url)
                except Exception:
                    continue

        print(f"\n✅ Finished crawling {len(visited)} pages. Stayed at last page: {last_url}")
        print("[RUN] Final page content snippet:")
        print(page.content()[:1000])

        input("\nPress ENTER to close browser...")
        browser.close()

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a multi-agent MCTS web agent or crawler.")
    parser.add_argument("--url", type=str, default="https://www.google.com", help="Start URL")
    parser.add_argument("--goal", type=str, default="News", help="Goal text to find on page")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--crawl", action="store_true", help="Crawl all links on the site instead of goal search")
    args = parser.parse_args()

    if args.crawl:
        run_web_crawler(args.url, headless=args.headless)
    else:
        run_web_multi_agent(args.url, args.goal, headless=args.headless)
