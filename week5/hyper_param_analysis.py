import json

import matplotlib.pyplot as plt

with open("./grid_search_eh_st_rr") as f:
    lines = json.load(f)

target = "eh5000"
target_lines = {}
for line in lines:
    if target in line:
        target_lines[line] = lines[line]

fig, ax = plt.subplots()
for line in target_lines.values():
    ax.plot(line)
plt.legend(lines.keys())
plt.show()
# ax.set_title(f"Iteration {n_iter}", fontsize=10, fontweight="bold")
# ax.set_xlim([0, self.width])
# ax.set_ylim([0, self.height])
# ax.set_xticks([])
# ax.set_yticks([])
# plt.savefig(self.save_path / f"ni{n_iter}.png")
# plt.close(fig)



