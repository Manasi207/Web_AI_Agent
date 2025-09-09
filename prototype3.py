# """
# web_multi_agent_mcts_prototype.py

# Prototype: Multi-Agent Web AI with MCTS variations:
#  - Open-Loop MCTS (states are action-sequence based clones)
#  - Progressive Widening
#  - Neural-guided priors (LLM or heuristic)
#  - Parallelized rollouts (thread pool)

# Built on top of the user's backbone script. This file provides a runnable prototype that
# demonstrates the variations requested. It keeps Playwright-based live execution for
# "act_live" but performs planning and rollouts on cloned snapshots (open-loop).

# Requirements:
#  - playwright (pip install playwright) and "playwright install"
#  - openai (optional, for real LLM priors). If not available, heuristic scoring used.


# """

import os
import time
import math
import random
import argparse
from typing import List, Optional, Any, Dict
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        """Return prior probability (0..1) that this action is promising.
        If real LLM available, query; otherwise use a light heuristic.
        """
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
            boost_keywords = ["search", "next", "submit", "go", "book", "confirm", "continue", "login", "find", "details", "more"]
            penalty_keywords = ["cancel", "close", "dismiss", "back"]
            for kw in boost_keywords:
                if kw in lower:
                    score += 0.15
            for kw in penalty_keywords:
                if kw in lower:
                    score -= 0.25
            # content-based tiny randomization so not always ties
            score = score + (random.random() - 0.5) * 0.05
            return max(0.0, min(1.0, score))

llm = LLMConnector()

# ---------------------------
# MCTS core classes
# ---------------------------
class ActionNode:
    def __init__(self, state: 'WebState', parent: Optional['ActionNode']=None, action: Optional[str]=None, prior: float=0.5):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[ActionNode] = []
        self.visits = 0
        self.value = 0.0
        self.prior = prior  # prior probability from neural/LLM guidance

    def is_fully_expanded(self, progressive_k: float = 1.0, progressive_alpha: float = 0.5) -> bool:
        """Progressive widening: allow |children| <= k * visits^alpha (rounded up)
        If visits == 0, allow at least 1 child.
        """
        limit = max(1, int(math.ceil(progressive_k * (self.visits ** progressive_alpha))))
        return len(self.children) >= limit

    def best_child(self, exploration_weight=1.0) -> 'ActionNode':
        # PUCT-style: Q + c_puct * p * sqrt(N)/ (1 + n)
        total_N = sum(c.visits for c in self.children) + 1
        def score(c):
            q = c.value / (c.visits + 1e-9)
            u = exploration_weight * c.prior * math.sqrt(total_N) / (1 + c.visits)
            return q + u
        return max(self.children, key=score)

# ---------------------------
# WebState abstraction (Open-loop: clones hold snapshots only)
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
        # lightweight open-loop simulation: remove the action and sometimes add synthetic successors
        actions = list(self.snapshot.get("actions", []))
        if action_text in actions:
            actions.remove(action_text)
        synthetic_new = []
        if random.random() < 0.35:
            synthetic_new.append(action_text + " → next")
        # also sometimes add a handful of other synthetic actions to increase branching (to demo PW)
        if random.random() < 0.15:
            for i in range(random.randint(0,2)):
                synthetic_new.append(f"auto_{random.randint(1,100)}")
        self.snapshot["actions"] = actions + synthetic_new
        # also mutate content slightly
        self.snapshot["content"] = (self.snapshot.get("content","") + "\n" + action_text)[:4000]
        return

    def clone(self) -> 'WebState':
        copied_snapshot = {"actions": list(self.snapshot.get("actions", [])),
                           "content": str(self.snapshot.get("content", ""))}
        return WebState(page=None, url=self.url, snapshot=copied_snapshot, live=False)

    def is_goal_state(self, goal_text: str) -> bool:
        return goal_text.lower() in self.snapshot.get("content", "").lower()

# ---------------------------
# Agents (Navigator/Extractor/Validator)
# ---------------------------
class BaseAgent:
    def __init__(self, name: str, llm: LLMConnector, rollout_depth: int = 3, iterations: int = 40,
                 progressive_k: float = 1.0, progressive_alpha: float = 0.5):
        self.name = name
        self.llm = llm
        self.rollout_depth = rollout_depth
        self.iterations = iterations
        # progressive widening parameters
        self.pw_k = progressive_k
        self.pw_alpha = progressive_alpha

    def suggest_local_actions(self, state: WebState) -> List[str]:
        return state.get_available_actions()

    def evaluate_goal(self, state: WebState, goal_text: str) -> float:
        return 1.0 if state.is_goal_state(goal_text) else 0.0

    def act_live(self, state: WebState, action_text: str) -> bool:
        return state.apply_action_live(action_text)

class NavigatorAgent(BaseAgent):
    def __init__(self, llm, rollout_depth=3, iterations=40):
        super().__init__("Navigator", llm, rollout_depth, iterations, progressive_k=1.2, progressive_alpha=0.6)

    def suggest_local_actions(self, state: WebState) -> List[str]:
        actions = state.get_available_actions()
        scored = [(a, self.llm.score_action(state.snapshot.get("content", ""), a)) for a in actions]
        scored_sorted = [a for a, s in sorted(scored, key=lambda x: -x[1])]
        return scored_sorted

class ExtractorAgent(BaseAgent):
    def __init__(self, llm, rollout_depth=2, iterations=30):
        super().__init__("Extractor", llm, rollout_depth, iterations, progressive_k=0.9, progressive_alpha=0.5)

    def suggest_local_actions(self, state: WebState) -> List[str]:
        actions = state.get_available_actions()
        priority = []
        for a in actions:
            if any(k in a.lower() for k in ["details", "view", "more", "info", "price", "select"]):
                priority.append(a)
        # fallback: use LLM priors to order remaining
        rest = [a for a in actions if a not in priority]
        scored = sorted(rest, key=lambda x: -self.llm.score_action(state.snapshot.get("content",""), x))
        return priority + scored

class ValidatorAgent(BaseAgent):
    def __init__(self, llm, rollout_depth=2, iterations=20):
        super().__init__("Validator", llm, rollout_depth, iterations, progressive_k=0.8, progressive_alpha=0.45)

    def evaluate_goal(self, state: WebState, goal_text: str) -> float:
        content = state.snapshot.get("content", "").lower()
        if goal_text.lower() in content:
            return 1.0
        if any(x in content for x in ["success", "confirmed", "booked", "thank you", "order received"]):
            return 0.9
        # ask LLM whether this is confirmation (returns 0..1)
        return self.llm.score_action(content, "is this a confirmation?")

# ---------------------------
# Local MCTS (with progressive widening, neural priors and threaded rollouts)
# ---------------------------
class LocalMCTS:
    def __init__(self, agent: BaseAgent, goal_text: str, exploration_weight: float = 1.0, parallel_rollouts: int = 8):
        self.agent = agent
        self.goal_text = goal_text
        self.exploration_weight = exploration_weight
        self.parallel_rollouts = parallel_rollouts

    def search(self, root_state: WebState) -> Optional[str]:
        root_node = ActionNode(root_state, prior=1.0)
        # initialize priors for root's candidate actions
        candidate_actions = self.agent.suggest_local_actions(root_state)
        for a in candidate_actions:
            p = self.agent.llm.score_action(root_state.snapshot.get("content", ""), a)
            new_state = root_state.clone()
            new_state.simulate_apply_action(a)
            child = ActionNode(new_state, parent=root_node, action=a, prior=p)
            root_node.children.append(child)

        # run iterations using threaded rollouts for speed
        for _ in range(self.agent.iterations):
            leaf = self._select(root_node)
            # run simulation(s) in parallel and average reward
            rewards = self._parallel_simulate(leaf, num=self.parallel_rollouts)
            reward = sum(rewards) / max(1, len(rewards))
            self._backpropagate(leaf, reward)

        # pick best child by highest visit count or value
        if not root_node.children:
            return None
        best = max(root_node.children, key=lambda c: (c.visits, c.value))
        return best.action

    def _select(self, node: ActionNode) -> ActionNode:
        # navigate until we can expand
        while True:
            if not node.children:
                return self._expand(node)
            if not node.is_fully_expanded(self.agent.pw_k, self.agent.pw_alpha):
                return self._expand(node)
            node = node.best_child(self.exploration_weight)
        return node

    def _expand(self, node: ActionNode) -> ActionNode:
        actions = self.agent.suggest_local_actions(node.state)
        # progressive widening: allow creating up to limit children
        for act in actions:
            if all(child.action != act for child in node.children):
                new_state = node.state.clone()
                new_state.simulate_apply_action(act)
                prior = self.agent.llm.score_action(node.state.snapshot.get("content", ""), act)
                child = ActionNode(new_state, parent=node, action=act, prior=prior)
                node.children.append(child)
                return child
        return node

    def _simulate_single(self, node: ActionNode) -> float:
        state = node.state.clone()
        for _ in range(self.agent.rollout_depth):
            actions = self.agent.suggest_local_actions(state)
            if not actions:
                break
            # use LLM-biased sampling in rollouts
            weights = [self.agent.llm.score_action(state.snapshot.get("content",""), a) + 1e-3 for a in actions]
            # normalize
            s = sum(weights)
            probs = [w / s for w in weights]
            act = random.choices(actions, probs)[0]
            state.simulate_apply_action(act)
            if state.is_goal_state(self.goal_text):
                return 1.0
        return self.agent.evaluate_goal(state, self.goal_text)

    def _parallel_simulate(self, node: ActionNode, num: int = 8) -> List[float]:
        results = []
        # thread-pool: safe since clones are independent and lightweight
        with ThreadPoolExecutor(max_workers=min(num, 8)) as ex:
            futures = [ex.submit(self._simulate_single, node) for _ in range(num)]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception:
                    results.append(0.0)
        return results

    def _backpropagate(self, node: ActionNode, reward: float):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

# ---------------------------
# Global Orchestrator MCTS (plans which agent to run) - uses similar ideas
# ---------------------------
class OrchestratorMCTS:
    def __init__(self, agents: List[BaseAgent], goal_text: str, iterations: int = 80, rollout_len: int = 6, exploration_weight: float = 1.0):
        self.agents = agents
        self.goal_text = goal_text
        self.iterations = iterations
        self.rollout_len = rollout_len
        self.exploration_weight = exploration_weight

    def plan(self, live_state: WebState) -> List[BaseAgent]:
        root = ActionNode(live_state.clone(), parent=None, action=None, prior=1.0)
        # initialize children as agents with priors computed from current snapshot
        for agent in self.agents:
            # compute a simple prior by asking agent to score "usefulness"
            # we take top suggested action's prior or default 0.5
            cand = agent.suggest_local_actions(root.state)
            prior = 0.5
            if cand:
                prior = agent.llm.score_action(root.state.snapshot.get("content",""), cand[0])
            new_state = root.state.clone()
            if cand:
                new_state.simulate_apply_action(cand[0])
            child = ActionNode(new_state, parent=root, action=agent.name, prior=prior)
            root.children.append(child)

        for _ in range(self.iterations):
            node = self._select(root)
            reward = self._simulate(node)
            self._backpropagate(node, reward)

        # build plan by selecting best children greedily
        plan = []
        node = root
        for _ in range(self.rollout_len):
            if not node.children:
                break
            child = node.best_child(exploration_weight=0.0)
            if child.action is None:
                break
            agent = next((a for a in self.agents if a.name == child.action), None)
            if agent:
                plan.append(agent)
            node = child
        return plan

    def _select(self, node: ActionNode) -> ActionNode:
        while True:
            if not node.children:
                return self._expand(node)
            if not node.is_fully_expanded(progressive_k=1.0, progressive_alpha=0.5):
                return self._expand(node)
            node = node.best_child(self.exploration_weight)
        return node

    def _expand(self, node: ActionNode) -> ActionNode:
        for agent in self.agents:
            if all(child.action != agent.name for child in node.children):
                new_state = node.state.clone()
                candidate_actions = agent.suggest_local_actions(new_state)
                if candidate_actions:
                    chosen = random.choice(candidate_actions)
                    new_state.simulate_apply_action(chosen)
                child = ActionNode(new_state, parent=node, action=agent.name,
                                   prior=agent.llm.score_action(node.state.snapshot.get("content",""), chosen if candidate_actions else ""))
                node.children.append(child)
                return child
        return node

    def _simulate(self, node: ActionNode) -> float:
        state = node.state.clone()
        total_reward = 0.0
        # run a short sequence of agent-guided rollouts
        for _ in range(self.rollout_len):
            agent = random.choice(self.agents)
            local = LocalMCTS(agent, self.goal_text, exploration_weight=1.0, parallel_rollouts=4)
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
# Runner - Multi-agent (live)
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

        navigator = NavigatorAgent(llm, rollout_depth=3, iterations=40)
        extractor = ExtractorAgent(llm, rollout_depth=2, iterations=30)
        validator = ValidatorAgent(llm, rollout_depth=2, iterations=20)
        agents = [navigator, extractor, validator]

        orchestrator = OrchestratorMCTS(agents, goal_text, iterations=80, rollout_len=6)

        plan = orchestrator.plan(live_state)
        print("[ORCH] Planned agent order:", [a.name for a in plan])

        for agent in plan:
            print(f"[EXEC] Running agent: {agent.name}")
            live_state.refresh_snapshot_from_live()
            local_mcts = LocalMCTS(agent, goal_text, exploration_weight=1.0, parallel_rollouts=6)
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
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a multi-agent MCTS web agent or crawler (prototype).")
    parser.add_argument("--url", type=str, default="https://www.google.com", help="Start URL")
    parser.add_argument("--goal", type=str, default="News", help="Goal text to find on page")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--crawl", action="store_true", help="Crawl all links on the site instead of goal search")
    args = parser.parse_args()

    if args.crawl:
        run_web_crawler(args.url, headless=args.headless)
    else:
        run_web_multi_agent(args.url, args.goal, headless=args.headless)

####################################################################################################################


# import argparse
# import random
# import math
# import asyncio
# from typing import List, Optional
# from playwright.sync_api import sync_playwright


# # -------------------------------
# # Web State Representation
# # -------------------------------
# class WebState:
#     def __init__(self, url: str, page=None):
#         self.url = url
#         self.page = page
#         self.actions = []
#         self.text = ""
#         if page:
#             self.refresh_snapshot_from_live()

#     def refresh_snapshot_from_live(self):
#         """Take a snapshot of current page elements and text."""
#         try:
#             self.text = self.page.inner_text("body")
#         except Exception:
#             self.text = ""
#         # Collect generic clickable text elements
#         buttons = self.page.query_selector_all("button")
#         links = self.page.query_selector_all("a")
#         inputs = self.page.query_selector_all("input[type=submit], input[type=button]")
#         self.actions = []
#         for el in buttons + links + inputs:
#             try:
#                 txt = el.inner_text().strip()
#                 if txt:
#                     self.actions.append(txt)
#             except Exception:
#                 pass
#         # Add Google-specific labels if available
#         if self.page.query_selector('a[href*="ServiceLogin"]'):
#             self.actions.append("Google Sign in")
#         if self.page.query_selector('a[href*="mail.google.com"]'):
#             self.actions.append("Gmail")
#         if self.page.query_selector('a[href*="imghp"]'):
#             self.actions.append("Images")
#         if self.page.query_selector('input[name="btnK"]'):
#             self.actions.append("Search")
#         if self.page.query_selector('input[name="btnI"]'):
#             self.actions.append("Feeling Lucky")

#     def available_actions(self) -> List[str]:
#         return self.actions

#     def apply_action_live(self, action_text: str, timeout_ms: int = 1500):
#         """Actually click or act on the live page using known selectors."""
#         if not self.page:
#             raise RuntimeError("No live page available for live action.")
#         try:
#             el = None
#             # --- Google specific buttons ---
#             if "Google Sign in" in action_text or "Sign in" in action_text:
#                 el = self.page.query_selector('a[href*="ServiceLogin"]')
#             elif "Gmail" in action_text:
#                 el = self.page.query_selector('a[href*="mail.google.com"]')
#             elif "Images" in action_text:
#                 el = self.page.query_selector('a[href*="imghp"]')
#             elif "Search" in action_text:
#                 el = self.page.query_selector('input[name="btnK"]')
#             elif "Lucky" in action_text or "Feeling Lucky" in action_text:
#                 el = self.page.query_selector('input[name="btnI"]')
#             # --- generic fallback ---
#             if not el:
#                 el = self.page.query_selector(f'button:has-text("{action_text}")') or \
#                      self.page.query_selector(f'a:has-text("{action_text}")') or \
#                      self.page.query_selector(f'input[value="{action_text}"]')
#             if el:
#                 el.click()
#                 self.page.wait_for_timeout(timeout_ms)
#                 self.refresh_snapshot_from_live()
#                 print(f"[LIVE] Clicked: {action_text}")
#                 return True
#             else:
#                 print(f"[LIVE] No live element matched: {action_text}")
#                 return False
#         except Exception as e:
#             print(f"[LIVE] Error clicking {action_text}: {e}")
#             return False


# # -------------------------------
# # MCTS Node
# # -------------------------------
# class Node:
#     def __init__(self, state: WebState, parent: Optional['Node'] = None, action: Optional[str] = None):
#         self.state = state
#         self.parent = parent
#         self.action = action
#         self.children: List[Node] = []
#         self.visits = 0
#         self.value = 0.0
#         self.untried_actions = state.available_actions()

#     def is_fully_expanded(self):
#         return len(self.untried_actions) == 0

#     def best_child(self, c_param=1.4):
#         best_score = -float("inf")
#         best_node = None
#         for child in self.children:
#             exploitation = child.value / (child.visits + 1e-6)
#             exploration = c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
#             score = exploitation + exploration
#             if score > best_score:
#                 best_score = score
#                 best_node = child
#         return best_node

#     def expand(self):
#         if not self.untried_actions:
#             return None
#         action = self.untried_actions.pop()
#         new_state = self.state
#         child = Node(new_state, parent=self, action=action)
#         self.children.append(child)
#         return child


# # -------------------------------
# # Simulation & Backprop
# # -------------------------------
# def rollout(state: WebState, goal: Optional[str] = None, max_depth: int = 5):
#     reward = 0.0
#     for _ in range(max_depth):
#         actions = state.available_actions()
#         if not actions:
#             break
#         action = random.choice(actions)
#         if goal and goal.lower() in action.lower():
#             reward += 1.0
#     return reward


# def backpropagate(node: Node, reward: float):
#     while node is not None:
#         node.visits += 1
#         node.value += reward
#         node = node.parent


# # -------------------------------
# # Monte Carlo Tree Search
# # -------------------------------
# def mcts(root: Node, iterations: int = 20, goal: Optional[str] = None):
#     for _ in range(iterations):
#         node = root
#         # Selection
#         while node.is_fully_expanded() and node.children:
#             node = node.best_child()
#         # Expansion
#         if not node.is_fully_expanded():
#             node = node.expand()
#         # Simulation
#         reward = rollout(node.state, goal=goal)
#         # Backpropagation
#         backpropagate(node, reward)
#     return root.best_child(c_param=0.0)


# # -------------------------------
# # Main Execution
# # -------------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--url", required=True)
#     parser.add_argument("--goal", default=None)
#     parser.add_argument("--crawl", action="store_true")
#     parser.add_argument("--headless", action="store_true")
#     args = parser.parse_args()

#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=args.headless)
#         page = browser.new_page()
#         page.goto(args.url)
#         state = WebState(args.url, page=page)
#         root = Node(state)

#         if args.crawl:
#             print("[MODE] Crawl entire site.")
#             for _ in range(3):
#                 best = mcts(root, iterations=15)
#                 if best and best.action:
#                     print(f"[CRAWL] Next action: {best.action}")
#                     state.apply_action_live(best.action)
#         else:
#             print(f"[MODE] Goal: {args.goal}")
#             best = mcts(root, iterations=20, goal=args.goal)
#             if best and best.action:
#                 print(f"[GOAL] Executing best action: {best.action}")
#                 state.apply_action_live(best.action)

#         browser.close()


# if __name__ == "__main__":
#     main()

# ## How to run : Instructions 
# ##   1] Run with a specific goal --> 
#      #     python prototype3.py --url "https://google.com" --crawl --headless
# #----------------------------------------------------------------------------------------------
# ##   2] Run as a crawler (no goal,just explore) -->
#      # python prototype3.py --url "https://example.com" --goal "News" --headless

# ##   3] Remove --headless if you wanna see the browser actions live