classes = [
    ("async_graph_bench.data_source", "DataSource"),
    ("async_graph_bench.manager", "BenchmarkManager"),
    ("async_graph_bench.stores", "DataStore"),
    ("async_graph_bench.stores.serializers", "Serializer"),
    ("async_graph_bench.models", "Model"),
    ("async_graph_bench", "SamplingConfig"),
    ("async_graph_bench", "NodeConfig"),
    ("async_graph_bench", "visualize_graph"),
    # add more (module, class) tuples here
    # e.g. ("async_graph_bench.executor", "Executor"),
]

import os

output_dir = "source/api"
os.makedirs(output_dir, exist_ok=True)

for module, cls in classes:
    filename = os.path.join(output_dir, f"{cls.lower()}.md")
    with open(filename, "w") as f:
        f.write(f"# `{cls}`\n\n")
        f.write("```{eval-rst}\n")
        f.write(f".. autoclass:: {module}.{cls}\n")
        f.write("   :members:\n")
        f.write("   :undoc-members:\n")
        f.write("   :show-inheritance:\n")
        f.write("```\n")
    print(f"✅ Wrote {filename}")
