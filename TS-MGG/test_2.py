import networkx as nx
import matplotlib.pyplot as plt

# 创建一个简单的网络
G = nx.Graph()
G.add_nodes_from([1, 2, 3])
G.add_edges_from([(1, 2), (2, 3)])

# 创建一个字典，包含每个节点的字体大小
node_font_sizes = {1: 12, 2: 16, 3: 20}
node_font_sizes = list(node_font_sizes.values())
print(node_font_sizes)
# 绘制网络和标签
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, font_size=node_font_sizes)

# 显示绘制的图形
plt.show()