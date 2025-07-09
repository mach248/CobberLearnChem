import numpy as np
import matplotlib.pyplot as plt

# Example Q-table from your training (reshape if needed)
Q = np.array([
    [0.529, 0.499, 0.491, 0.512],
    [0.306, 0.322, 0.304, 0.481],
    [0.431, 0.410, 0.375, 0.448],
    [0.345, 0.242, 0.310, 0.433],
    [0.553, 0.164, 0.329, 0.416],
    [0.000, 0.000, 0.000, 0.000],
    [0.343, 0.106, 0.219, 0.084],
    [0.000, 0.000, 0.000, 0.000],
    [0.457, 0.335, 0.437, 0.589],
    [0.410, 0.656, 0.448, 0.441],
    [0.732, 0.350, 0.365, 0.371],
    [0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000],
    [0.514, 0.547, 0.716, 0.604],
    [0.706, 0.858, 0.758, 0.762],
    [0.000, 0.000, 0.000, 0.000]
])

# Arrow directions for each action
# 0: Left, 1: Down, 2: Right, 3: Up
arrow_dict = {
    0: (-1, 0),  # Left
    1: (0, -1),  # Down
    2: (1, 0),  # Right
    3: (0, 1)  # Up
}

plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)
ax.set_xticks(np.arange(0, 4))
ax.set_yticks(np.arange(0, 4))
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.grid(True)

for state in range(16):
    if np.all(Q[state] == 0):
        continue  # skip unexplored or terminal states

    best_action = np.argmax(Q[state])
    dx, dy = arrow_dict[best_action]

    # Convert state to 2D grid position (col, row)
    x = state % 4
    y = 3 - (state // 4)  # Flip y for proper orientation

    # Draw arrow
    plt.arrow(x, y, 0.3 * dx, 0.3 * dy, head_width=0.1, color='blue')

# Add state numbers (optional)
for state in range(16):
    x = state % 4
    y = 3 - (state // 4)
    plt.text(x, y, str(state), va='center', ha='center', fontsize=8)

plt.title("FrozenLake Q-table Policy (Best Action per State)")
plt.tight_layout()
plt.show()
