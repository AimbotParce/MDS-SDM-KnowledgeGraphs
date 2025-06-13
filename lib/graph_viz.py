import io

import matplotlib.pyplot as plt
import pydotplus
from rdflib import Graph
from rdflib.tools.rdf2dot import rdf2dot


def visualize_rdfs(g: Graph):
    stream = io.StringIO()
    rdf2dot(g, stream)
    dg = pydotplus.graph_from_dot_data(stream.getvalue())
    png = dg.create_png()
    plt.figure(figsize=(12, 12))
    plt.imshow(plt.imread(io.BytesIO(png)))
    plt.axis("off")
    plt.show()


def save_rdfs_visualization(g: Graph, filename: str):
    stream = io.StringIO()
    rdf2dot(g, stream)
    dg = pydotplus.graph_from_dot_data(stream.getvalue())
    png = dg.create_png()
    with open(filename, "wb") as f:
        f.write(png)
