import networkx as nx

from Code.constants import CRAWLER_OUTPUT_JSON, PAGE_RANK_ALPHA
import json

with open(CRAWLER_OUTPUT_JSON) as json_file:
    pages = json.load(json_file)

counter = 0
for page_id in pages.keys():
    pages[page_id]["count"] = counter
    counter += 1

# print(pages)
G = nx.Graph()
# print(len(pages))
for page_id in pages:
    # print(pages[page_id])
    page = pages[page_id]
    if not G.has_node(page["count"]):
        G.add_node(page["count"])
    for ref in page["references"]:
        if ref in pages:
            if not G.has_node(pages[ref]["count"]):
                G.add_node(pages[ref]["count"])
            G.add_edge(page["count"], pages[ref]["count"])
            # print("Edge added for", page["count"], pages[ref]["count"])
page_rank = nx.pagerank(G, PAGE_RANK_ALPHA)
print(page_rank)
