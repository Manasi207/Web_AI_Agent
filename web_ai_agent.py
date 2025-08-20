
#######################################################################################

# import time
# import math
# import random
# from typing import List
# from playwright.sync_api import sync_playwright


# # Action node for MCTS
# class ActionNode:
#     def __init__(self, state, parent=None, action=None):
#         self.state = state
#         self.parent = parent
#         self.action = action
#         self.children = []
#         self.visits = 0
#         self.value = 0.0

#     def is_fully_expanded(self):
#         return len(self.children) >= len(self.state.get_available_actions())

#     def best_child(self, exploration_weight=1.41):
#         return max(self.children, key=lambda c: (c.value / (c.visits + 1e-5)) +
#                    exploration_weight * ((2 * math.log(self.visits + 1) / (c.visits + 1e-5)) ** 0.5))


# # Web state representation
# class WebState:
#     def __init__(self, page, url):
#         self.page = page
#         self.url = url

#     def get_available_actions(self) -> List[str]:
#         buttons = self.page.query_selector_all('button, a')
#         actions = []
#         for b in buttons:
#             try:
#                 if b.is_visible():
#                     html = b.get_attribute('outerHTML')
#                     if html:
#                         actions.append(html[:60])  # Truncated for simplicity
#             except:
#                 continue
#         return actions

#     def apply_action(self, action_selector: str):
#         try:
#             el = self.page.query_selector(f'xpath=//button[contains(., "{action_selector}")]') \
#                  or self.page.query_selector(f'xpath=//a[contains(., "{action_selector}")]')
#             if el:
#                 el.click()
#                 time.sleep(1)
#                 print(f"[+] Clicked on: {action_selector}")
#             else:
#                 print(f"[!] No matching element found for: {action_selector}")
#         except Exception as e:
#             print(f"[!] Error during action '{action_selector}': {e}")

#     def clone(self):
#         return self  # Placeholder: would deep-copy in a full implementation

#     def is_goal_state(self, goal_text: str) -> bool:
#         return goal_text.lower() in self.page.content().lower()


# # MCTS-based Agent
# class MCTSAgent:
#     def __init__(self, goal_text: str, iterations=30):
#         self.goal_text = goal_text
#         self.iterations = iterations

#     def search(self, root: ActionNode):
#         for _ in range(self.iterations):
#             node = self.select(root)
#             reward = self.simulate(node)
#             self.backpropagate(node, reward)
#         return root.best_child(exploration_weight=0)

#     def select(self, node: ActionNode) -> ActionNode:
#         while node.is_fully_expanded() and node.children:
#             node = node.best_child()
#         return self.expand(node)

#     def expand(self, node: ActionNode) -> ActionNode:
#         actions = node.state.get_available_actions()
#         for act in actions:
#             if all(child.action != act for child in node.children):
#                 new_state = node.state.clone()
#                 new_state.apply_action(act)
#                 child = ActionNode(new_state, parent=node, action=act)
#                 node.children.append(child)
#                 return child
#         return node

#     def simulate(self, node: ActionNode) -> float:
#         for _ in range(3):
#             actions = node.state.get_available_actions()
#             if not actions:
#                 break
#             act = random.choice(actions)
#             node.state.apply_action(act)
#             if node.state.is_goal_state(self.goal_text):
#                 return 1.0
#         return 0.0

#     def backpropagate(self, node: ActionNode, reward: float):
#         while node:
#             node.visits += 1
#             node.value += reward
#             node = node.parent


# # Entrypoint
# def run_web_agent(start_url: str, goal_text: str):
#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=False)
#         page = browser.new_page()
#         page.goto(start_url)
#         time.sleep(60)  # Ensure the page loads completely

#         root_state = WebState(page, start_url)
#         root_node = ActionNode(state=root_state)

#         agent = MCTSAgent(goal_text=goal_text, iterations=30)
#         best_node = agent.search(root_node)

#         print(f"\n[✓] Final chosen action: {best_node.action}")
#         print(f"[✓] Value: {best_node.value:.2f}, Visits: {best_node.visits}")
#         browser.close()


# # Example
# if __name__ == "__main__":
#     run_web_agent("https://example.com", "contact")
    
########################################################################
import time
import math
import random
from typing import List, Optional
from playwright.sync_api import sync_playwright


# ----------- Action Node for Monte Carlo Tree Search -----------
class ActionNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) >= len(self.state.get_available_actions())

    def best_child(self, exploration_weight=1.41):
        return max(
            self.children,
            key=lambda c: (c.value / (c.visits + 1e-5)) +
            exploration_weight * ((2 * math.log(self.visits + 1) / (c.visits + 1e-5)) ** 0.5)
        )


# ----------- Web Page State Representation -----------
class WebState:
    def __init__(self, page, url):
        self.page = page
        self.url = url

    def get_available_actions(self) -> List[str]:
        buttons = self.page.query_selector_all("button, a")
        actions = []
        for b in buttons:
            try:
                if b.is_visible():
                    text = b.inner_text().strip()
                    if text:
                        actions.append(text)
            except Exception:
                continue
        return actions

    def apply_action(self, action_text: str):
        try:
            el = self.page.query_selector(f'button:has-text("{action_text}")') or \
                 self.page.query_selector(f'a:has-text("{action_text}")')
            if el:
                el.click()
                self.page.wait_for_timeout(1500)  # 1.5 sec pause
                print(f"[+] Clicked on: {action_text}")
            else:
                print(f"[!] No element matched for: {action_text}")
        except Exception as e:
            print(f"[!] Error on action '{action_text}': {e}")

    def clone(self):
        return self  # placeholder for future replay-based state cloning

    def is_goal_state(self, goal_text: str) -> bool:
        return goal_text.lower() in self.page.content().lower()


# ----------- Monte Carlo Tree Search Agent -----------
class MCTSAgent:
    def __init__(self, goal_text: str, iterations=40):
        self.goal_text = goal_text
        self.iterations = iterations

    def search(self, root: ActionNode):
        for _ in range(self.iterations):
            node = self.select(root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        return root.best_child(exploration_weight=0)

    def select(self, node: ActionNode) -> ActionNode:
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return self.expand(node)

    def expand(self, node: ActionNode) -> ActionNode:
        actions = node.state.get_available_actions()
        for act in actions:
            if all(child.action != act for child in node.children):
                new_state = node.state.clone()
                new_state.apply_action(act)
                child = ActionNode(new_state, parent=node, action=act)
                node.children.append(child)
                return child
        return node

    def simulate(self, node: ActionNode) -> float:
        for _ in range(3):
            actions = node.state.get_available_actions()
            if not actions:
                break
            act = random.choice(actions)
            node.state.apply_action(act)
            if node.state.is_goal_state(self.goal_text):
                return 1.0
        return 0.0

    def backpropagate(self, node: ActionNode, reward: float):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent


# ----------- Entry Function for Running Agent -----------
def run_web_agent(start_url: str, goal_text: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(start_url, timeout=15000)
        page.wait_for_load_state("load")

        print(f"[🌐] Loaded: {start_url}")
        root_state = WebState(page, start_url)
        root_node = ActionNode(state=root_state)

        agent = MCTSAgent(goal_text=goal_text, iterations=30)
        best_node = agent.search(root_node)

        print(f"\n[✓] Best Action Chosen: {best_node.action}")
        print(f"[✓] Value: {best_node.value:.2f}, Visits: {best_node.visits}")
        browser.close()


# ----------- Run Agent on Site -----------
if __name__ == "__main__":
    run_web_agent("https://google.com", "News")

