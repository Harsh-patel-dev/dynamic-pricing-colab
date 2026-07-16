import random
import math

class Market:
    def __init__(self, market_type="monopoly", base_demand=100.0, price_sensitivity=2.5, noise_std=5.0, buyer_model="linear"):
        """
        market_type: "monopoly" (1 player) or "oligopoly" (multiple players competing)
        buyer_model: "linear" (Q = A - B*P) or "logistic" (Discrete Choice MNL)
        """
        self.market_type = market_type
        self.base_demand = float(base_demand)
        self.price_sensitivity = float(price_sensitivity)
        self.noise_std = float(noise_std)
        self.buyer_model = buyer_model
        
    def simulate_demand(self, prices, cost=10.0):
        """
        prices: dict of {agent_id: price}
        Returns: dict of {agent_id: demand}
        """
        demands = {}
        if not prices:
            return demands

        if self.market_type == "monopoly":
            # Only focus on the first agent (or treat each independently if multiple)
            for agent_id, price in prices.items():
                if self.buyer_model == "linear":
                    # Q = A - B * P + noise
                    noise = random.gauss(0, self.noise_std)
                    demand = self.base_demand - self.price_sensitivity * price + noise
                    demands[agent_id] = max(0, int(round(demand)))
                else:
                    # Logistic / MNL choice probability for monopoly
                    # Probability that a single buyer purchases at price P
                    # Utility of buying: U = V - B * P. Utility of not buying: U0 = 0
                    # Prob = e^(V - B*P) / (1 + e^(V - B*P))
                    val_threshold = self.base_demand / self.price_sensitivity # surrogate for value
                    utility = self.price_sensitivity * (val_threshold - price)
                    try:
                        prob = 1.0 / (1.0 + math.exp(-utility))
                    except OverflowError:
                        prob = 0.0 if utility < 0 else 1.0
                    
                    # Say there are N potential buyers
                    potential_buyers = 100
                    successes = 0
                    for _ in range(potential_buyers):
                        if random.random() < prob:
                            successes += 1
                    demands[agent_id] = successes
        else:
            # Oligopoly - Competition using Multinomial Logit Model (MNL)
            # Consumers choose between all agents and an outside option (not buying)
            # Utility of agent i: U_i = V - B * P_i + epsilon_i
            # Utility of outside option: U_0 = V_0
            # Let's define V = 50.0 (constant value of the product)
            # Outside option utility V_0 = 10.0 (depends on base_demand)
            V = self.base_demand
            V_0 = 15.0  # Outside option utility
            
            # Calculate utility exponents
            exps = {}
            sum_exp = math.exp(V_0)
            
            for agent_id, price in prices.items():
                # Utility = V - B * P
                u = V - self.price_sensitivity * price
                try:
                    exp_u = math.exp(u)
                except OverflowError:
                    exp_u = 1e10 if u > 0 else 1e-10
                exps[agent_id] = exp_u
                sum_exp += exp_u
                
            # Total potential market size
            market_size = 150
            demands = {agent_id: 0 for agent_id in prices}
            
            # Simulate choice for each consumer
            for _ in range(market_size):
                r = random.random() * sum_exp
                current_sum = math.exp(V_0) # start with outside option
                if r < current_sum:
                    continue # Buyer chose not to buy
                
                for agent_id, exp_u in exps.items():
                    current_sum += exp_u
                    if r < current_sum:
                        demands[agent_id] += 1
                        break
                        
        return demands


class BaseAgent:
    def __init__(self, agent_id, cost=10.0, min_price=5.0, max_price=50.0):
        self.agent_id = agent_id
        self.cost = float(cost)
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.revenue_history = []
        self.profit_history = []
        self.price_history = []
        self.demand_history = []

    def get_price(self, state):
        raise NotImplementedError

    def update(self, state, price, demand, profit, next_state):
        self.price_history.append(price)
        self.demand_history.append(demand)
        self.revenue_history.append(price * demand)
        self.profit_history.append(profit)

    def get_total_profit(self):
        return sum(self.profit_history)
        
    def get_total_revenue(self):
        return sum(self.revenue_history)


class ConstantPriceAgent(BaseAgent):
    def __init__(self, agent_id, cost=10.0, price=20.0, **kwargs):
        super().__init__(agent_id, cost, **kwargs)
        self.price = float(price)

    def get_price(self, state):
        return min(self.max_price, max(self.min_price, self.price))


class CompetitorMatchingAgent(BaseAgent):
    def __init__(self, agent_id, cost=10.0, undercut_pct=0.05, **kwargs):
        super().__init__(agent_id, cost, **kwargs)
        self.undercut_pct = float(undercut_pct)

    def get_price(self, state):
        competitor_prices = state.get("competitor_prices", [])
        if not competitor_prices:
            # No competitors known, charge mid-price
            return (self.min_price + self.max_price) / 2.0
            
        lowest_comp = min(competitor_prices)
        # Undercut competitor
        target_price = lowest_comp * (1.0 - self.undercut_pct)
        # Never price below cost + a tiny margin
        floor_price = self.cost * 1.05
        return min(self.max_price, max(floor_price, target_price))


class MarkdownAgent(BaseAgent):
    def __init__(self, agent_id, cost=10.0, **kwargs):
        super().__init__(agent_id, cost, **kwargs)

    def get_price(self, state):
        # High price at the beginning, drops as time runs out or inventory remains high
        remaining_rounds = state.get("remaining_rounds", 10)
        total_rounds = state.get("total_rounds", 100)
        remaining_inventory = state.get("inventory", 100)
        initial_inventory = state.get("initial_inventory", 100)
        
        time_elapsed_ratio = (total_rounds - remaining_rounds) / max(1, total_rounds)
        inventory_sold_ratio = (initial_inventory - remaining_inventory) / max(1, initial_inventory)
        
        # If we are selling slower than time is expiring, lower the price
        # Markdown dynamic factor
        if time_elapsed_ratio > 0:
            sales_pace_ratio = inventory_sold_ratio / time_elapsed_ratio
        else:
            sales_pace_ratio = 1.0
            
        # Target price decreases over time and decreases if sales are slow
        base_target = self.max_price - (self.max_price - self.cost) * time_elapsed_ratio
        
        if sales_pace_ratio < 0.8:
            # Selling too slow, markdown further
            base_target *= 0.85
        elif sales_pace_ratio > 1.2:
            # Selling fast, raise price to milk the market
            base_target *= 1.1
            
        return min(self.max_price, max(self.min_price, base_target))


class LinearRegressionAgent(BaseAgent):
    def __init__(self, agent_id, cost=10.0, forgetting_factor=0.95, **kwargs):
        super().__init__(agent_id, cost, **kwargs)
        self.forgetting_factor = forgetting_factor
        
        # RLS parameters for: Q = alpha + beta * P (monopoly)
        # In case of competition: Q = alpha + beta * P + gamma * P_comp
        # We start with a simple linear demand model: Q = alpha + beta * P
        # Parameter vector theta = [alpha, beta]^T
        # Initial priors: alpha = 100 (decent demand), beta = -2.0 (negative slope)
        self.theta = [100.0, -2.0]
        # Covariance matrix P_cov = 1000 * I
        self.P_cov = [[1000.0, 0.0],
                      [0.0, 1000.0]]
        
    def get_price(self, state):
        # Optimal price for Q = alpha + beta * P:
        # Profit = (P - cost) * (alpha + beta * P)
        # dProfit/dP = alpha + 2 * beta * P - beta * cost = 0
        # P* = (beta * cost - alpha) / (2 * beta)
        alpha, beta = self.theta[0], self.theta[1]
        
        # If beta is positive (which violates demand theory), use fallback price
        if beta >= 0:
            price = self.cost * 1.5
        else:
            price = (beta * self.cost - alpha) / (2.0 * beta)
            
        # Add a tiny exploration noise (epsilon-greedy style) to learn the curve
        if random.random() < 0.1:
            price += random.uniform(-3.0, 3.0)
            
        return min(self.max_price, max(self.min_price, price))

    def update(self, state, price, demand, profit, next_state):
        super().update(state, price, demand, profit, next_state)
        
        # RLS Update:
        # x_t = [1.0, price]
        x = [1.0, price]
        
        # P_cov * x
        Px = [
            self.P_cov[0][0] * x[0] + self.P_cov[0][1] * x[1],
            self.P_cov[1][0] * x[0] + self.P_cov[1][1] * x[1]
        ]
        
        # x^T * P_cov * x
        xPx = x[0] * Px[0] + x[1] * Px[1]
        
        # Denominator: lambda + x^T * P_cov * x
        denom = self.forgetting_factor + xPx
        
        # Gain vector K = P_cov * x / denom
        K = [Px[0] / denom, Px[1] / denom]
        
        # Prediction error: e = Q - x^T * theta
        pred_q = self.theta[0] * x[0] + self.theta[1] * x[1]
        err = demand - pred_q
        
        # Update theta: theta = theta + K * err
        self.theta[0] += K[0] * err
        self.theta[1] += K[1] * err
        
        # Update covariance: P_cov = (P_cov - K * x^T * P_cov) / lambda
        # K * x^T is a 2x2 matrix:
        # [ K[0]*x[0], K[0]*x[1] ]
        # [ K[1]*x[0], K[1]*x[1] ]
        # K * x^T * P_cov is:
        # Row 0: K[0]*x[0]*P00 + K[0]*x[1]*P10, K[0]*x[0]*P01 + K[0]*x[1]*P11
        # Row 1: K[1]*x[0]*P00 + K[1]*x[1]*P10, K[1]*x[0]*P01 + K[1]*x[1]*P11
        
        k_x_P = [
            [K[0] * (x[0]*self.P_cov[0][0] + x[1]*self.P_cov[1][0]), K[0] * (x[0]*self.P_cov[0][1] + x[1]*self.P_cov[1][1])],
            [K[1] * (x[0]*self.P_cov[0][0] + x[1]*self.P_cov[1][0]), K[1] * (x[0]*self.P_cov[0][1] + x[1]*self.P_cov[1][1])]
        ]
        
        # Update and divide by forgetting factor lambda
        for i in range(2):
            for j in range(2):
                self.P_cov[i][j] = (self.P_cov[i][j] - k_x_P[i][j]) / self.forgetting_factor


class ThompsonSamplingAgent(BaseAgent):
    def __init__(self, agent_id, cost=10.0, num_price_arms=11, **kwargs):
        super().__init__(agent_id, cost, **kwargs)
        # Discretize pricing space into arms
        self.num_arms = num_price_arms
        # Generate price arms evenly between min_price and max_price
        self.arms = [self.min_price + i * (self.max_price - self.min_price) / (num_price_arms - 1) for i in range(num_price_arms)]
        
        # Beta priors for conversion rate at each price: (alphas, betas)
        # alpha = successes (sales), beta = failures (no sales)
        # We start with alpha=1, beta=1 (uniform distribution)
        self.alphas = [1.0] * num_arms
        self.betas = [1.0] * num_arms
        
        # Keep track of active price index chosen in the round
        self.last_arm_idx = 0

    def get_price(self, state):
        # Sample from the Beta distribution for each arm
        best_expected_profit = -9999.0
        best_arm_idx = 0
        
        for i in range(self.num_arms):
            # Sample conversion rate probability
            prob_sample = random.betavariate(self.alphas[i], self.betas[i])
            # Expected profit = (Price - Cost) * Conversion_probability
            expected_profit = (self.arms[i] - self.cost) * prob_sample
            
            if expected_profit > best_expected_profit:
                best_expected_profit = expected_profit
                best_arm_idx = i
                
        self.last_arm_idx = best_arm_idx
        return self.arms[best_arm_idx]

    def update(self, state, price, demand, profit, next_state):
        super().update(state, price, demand, profit, next_state)
        
        # We update the specific arm that was chosen.
        # Demand represents number of units sold.
        # We can view the customer choices as Bernoulli trials.
        # If we sell demand units, we have 'demand' successes.
        # But we need a denominator: the total potential market size (or max possible sales).
        # Let's assume a batch size (max potential buyers per agent) of 50.
        market_trials = max(50.0, demand * 1.5)
        
        successes = float(demand)
        failures = max(0.0, market_trials - successes)
        
        # Update Beta parameters for the chosen arm
        self.alphas[self.last_arm_idx] += successes
        self.betas[self.last_arm_idx] += failures


class QLearningAgent(BaseAgent):
    def __init__(self, agent_id, cost=10.0, learning_rate=0.2, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, **kwargs):
        super().__init__(agent_id, cost, **kwargs)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        # Discretize actions: 5 price levels
        self.num_actions = 5
        self.actions = [self.min_price + i * (self.max_price - self.min_price) / (self.num_actions - 1) for i in range(self.num_actions)]
        
        # Discretize states:
        # Time remaining: 0 (0-33%), 1 (34-66%), 2 (67-100%)
        # Inventory remaining: 0 (low < 30%), 1 (med 30-70%), 2 (high > 70%)
        # Competitor pricing state: 0 (we are cheapest), 1 (same), 2 (we are expensive), 3 (no competitor)
        # Total states: 3 * 3 * 4 = 36 states
        # Q-table represented as a dict: key is (time_state, inv_state, comp_state), value is list of Q-values for actions
        self.q_table = {}
        self.last_state_key = None
        self.last_action_idx = 0

    def _get_state_key(self, state):
        # Time state
        rem_rounds = state.get("remaining_rounds", 100)
        tot_rounds = state.get("total_rounds", 100)
        time_ratio = rem_rounds / max(1, tot_rounds)
        if time_ratio < 0.33:
            time_state = 0
        elif time_ratio < 0.66:
            time_state = 1
        else:
            time_state = 2
            
        # Inventory state
        inv = state.get("inventory", 100)
        init_inv = state.get("initial_inventory", 100)
        inv_ratio = inv / max(1, init_inv)
        if inv_ratio < 0.3:
            inv_state = 0
        elif inv_ratio < 0.7:
            inv_state = 1
        else:
            inv_state = 2
            
        # Competitor pricing state
        comp_prices = state.get("competitor_prices", [])
        if not comp_prices:
            comp_state = 3
        else:
            min_comp = min(comp_prices)
            # Find what price we chose in the last step (or average)
            my_last_price = self.price_history[-1] if self.price_history else (self.min_price + self.max_price)/2.0
            if my_last_price < min_comp - 1.0:
                comp_state = 0 # we are cheaper
            elif my_last_price > min_comp + 1.0:
                comp_state = 2 # we are expensive
            else:
                comp_state = 1 # same
                
        return (time_state, inv_state, comp_state)

    def get_price(self, state):
        state_key = self._get_state_key(state)
        self.last_state_key = state_key
        
        # Initialize Q-values for state if not exists
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.num_actions
            
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.num_actions - 1)
        else:
            q_values = self.q_table[state_key]
            # Argmax with tie-breaking
            max_q = max(q_values)
            indices = [i for i, q in enumerate(q_values) if q == max_q]
            action_idx = random.choice(indices)
            
        self.last_action_idx = action_idx
        return self.actions[action_idx]

    def update(self, state, price, demand, profit, next_state):
        super().update(state, price, demand, profit, next_state)
        
        state_key = self.last_state_key
        if state_key is None:
            return
            
        next_state_key = self._get_state_key(next_state)
        
        # Initialize Q-values for next state if not exists
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * self.num_actions
            
        # Reward function: Profit obtained
        # We can also add a penalty if inventory is exhausted or remains too high at the end
        reward = profit
        
        # Q-learning Update
        max_next_q = max(self.q_table[next_state_key])
        old_q = self.q_table[state_key][self.last_action_idx]
        
        self.q_table[state_key][self.last_action_idx] += self.lr * (reward + self.gamma * max_next_q - old_q)
        
        # Decay epsilon
        self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)


class Simulation:
    def __init__(self, agent_configs, market_config, total_rounds=100, initial_inventory=100):
        """
        agent_configs: list of dicts, e.g. [{"type": "q_learning", "id": "RL-1", "cost": 10.0}, ...]
        market_config: dict of market parameters
        """
        self.market = Market(**market_config)
        self.total_rounds = total_rounds
        self.initial_inventory = initial_inventory
        self.current_round = 0
        
        # Initialize agents
        self.agents = {}
        self.inventories = {}
        for config in agent_configs:
            a_type = config["type"]
            a_id = config["id"]
            cost = config.get("cost", 10.0)
            min_price = config.get("min_price", 5.0)
            max_price = config.get("max_price", 50.0)
            
            # Additional parameters
            kwargs = {"cost": cost, "min_price": min_price, "max_price": max_price}
            if a_type == "q_learning":
                kwargs["learning_rate"] = config.get("learning_rate", 0.2)
                kwargs["epsilon"] = config.get("epsilon", 1.0)
                agent = QLearningAgent(a_id, **kwargs)
            elif a_type == "thompson_sampling":
                agent = ThompsonSamplingAgent(a_id, **kwargs)
            elif a_type == "linear_regression":
                agent = LinearRegressionAgent(a_id, **kwargs)
            elif a_type == "competitor_matching":
                agent = CompetitorMatchingAgent(a_id, **kwargs)
            elif a_type == "markdown":
                agent = MarkdownAgent(a_id, **kwargs)
            else:
                agent = ConstantPriceAgent(a_id, price=config.get("price", 20.0), **kwargs)
                
            self.agents[a_id] = agent
            self.inventories[a_id] = initial_inventory
            
        self.history = []

    def get_state(self, agent_id):
        # Compile state dictionary for a specific agent
        competitor_prices = []
        # Look at the last price chosen by other agents
        for other_id, agent in self.agents.items():
            if other_id != agent_id and agent.price_history:
                competitor_prices.append(agent.price_history[-1])
                
        return {
            "remaining_rounds": self.total_rounds - self.current_round,
            "total_rounds": self.total_rounds,
            "inventory": self.inventories[agent_id],
            "initial_inventory": self.initial_inventory,
            "competitor_prices": competitor_prices
        }

    def step(self):
        if self.current_round >= self.total_rounds:
            return None

        # 1. Get prices from all agents
        prices = {}
        states = {}
        for agent_id, agent in self.agents.items():
            # If agent is out of stock, they cannot price / participate
            if self.inventories[agent_id] <= 0:
                continue
            states[agent_id] = self.get_state(agent_id)
            prices[agent_id] = agent.get_price(states[agent_id])

        # 2. Simulate demand
        demands = self.market.simulate_demand(prices)

        # 3. Apply capacity limits & calculate profit
        step_results = {
            "round": self.current_round + 1,
            "prices": {},
            "demands": {},
            "sales": {},
            "profits": {},
            "inventories": {}
        }
        
        next_states = {}
        
        for agent_id, agent in self.agents.items():
            # If agent was out of stock already
            if self.inventories[agent_id] <= 0:
                step_results["prices"][agent_id] = None
                step_results["demands"][agent_id] = 0
                step_results["sales"][agent_id] = 0
                step_results["profits"][agent_id] = 0.0
                step_results["inventories"][agent_id] = 0
                continue
                
            price = prices[agent_id]
            demand = demands.get(agent_id, 0)
            
            # Sales cannot exceed current inventory
            sales = min(demand, self.inventories[agent_id])
            self.inventories[agent_id] -= sales
            
            # Profit = Sales * (Price - Cost)
            profit = float(sales) * (price - agent.cost)
            
            step_results["prices"][agent_id] = price
            step_results["demands"][agent_id] = demand
            step_results["sales"][agent_id] = sales
            step_results["profits"][agent_id] = profit
            step_results["inventories"][agent_id] = self.inventories[agent_id]
            
            # Get next state
            next_states[agent_id] = self.get_state(agent_id)
            
            # Update agent model
            agent.update(states[agent_id], price, sales, profit, next_states[agent_id])

        self.current_round += 1
        self.history.append(step_results)
        return step_results

    def run_all(self):
        results = []
        while self.current_round < self.total_rounds:
            res = self.step()
            if res:
                results.append(res)
        return results
