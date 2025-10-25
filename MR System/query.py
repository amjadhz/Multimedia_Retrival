# query_example.py

from step4 import mesh_querying

# 查询
classes, results = mesh_querying(
    model_file_name="../data/Guitar/D00023.obj",
    csv_path="database/all_features.csv",
    distance_stats_path="database/distance_stats.csv",
    K=10
)

print("\nresults for data/Guitar/D00023.obj:")
print("=" * 70)
for i, ((cls, fname), dist) in enumerate(results, 1):
    print(f"{i:2}. {cls:15} | {fname:30} | distance: {dist:8.4f}")

print(f"\nTop-10 class: {classes}")