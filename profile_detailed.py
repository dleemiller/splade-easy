import sys
import cProfile
import pstats
from io import StringIO
from pathlib import Path
from splade_easy import IndexReader

# Load index
index_reader = IndexReader(Path(sys.argv[1]))

# Run one query to warm up cache
query = "machine learning algorithms"
index_reader.search(query, top_k=3, return_text=False)

# Profile second query (cache is warm)
pr = cProfile.Profile()
pr.enable()
for _ in range(10):
    results = index_reader.search(query, top_k=3, return_text=False)
pr.disable()

# Show detailed stats
s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(50)
print(s.getvalue())
