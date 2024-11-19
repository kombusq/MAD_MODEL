import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv('/Users/tarun/finanical eng./data fe 3.csv')
data.set_index(data.columns[0], inplace=True)  
data = data.apply(pd.to_numeric, errors='coerce').dropna()

# Step 2: Select 20 random stocks
selected_stocks = data.sample(n=20, axis=1, random_state=42)

# Step 3: Calculate returns
returns = selected_stocks.pct_change().dropna()

# Step 4: Define the MAD model
num_stocks = returns.shape[1]
weights = cp.Variable(num_stocks)
expected_return = cp.Parameter(nonneg=True)
absolute_deviation = cp.Variable(len(returns))

# Objective: Minimize Mean Absolute Deviation
objective = cp.Minimize(cp.sum(absolute_deviation))

# Constraints
constraints = [
    cp.sum(weights) == 1,  
    weights >= 0,         
    returns.values @ weights >= expected_return, 
    absolute_deviation >= returns.values @ weights - expected_return,
    absolute_deviation >= expected_return - returns.values @ weights
]

# Set up the problem
prob = cp.Problem(objective, constraints)

# Step 5: Solve for 10 expected returns and store results
efficient_portfolios = []
expected_returns = [0.05 + i * 0.01 for i in range(10)]  # From 5% to 5.09%

for r in expected_returns:
    expected_return.value = r
    prob.solve()
    risk = prob.value
    portfolio_weights = weights.value
    efficient_portfolios.append((r, risk, portfolio_weights))

# Step 6: Plot the efficient frontier
returns, risks = zip(*[(r, risk) for r, risk, _ in efficient_portfolios])
plt.figure(figsize=(10, 6))
plt.plot(risks, returns, marker='o')
plt.xlabel('Risk (MAD)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.grid()
plt.show()
