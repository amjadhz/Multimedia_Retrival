from step4 import mesh_querying

classes, results = mesh_querying(
    #model_file_name="../data/AquaticAnimal/m80.obj",
    model_file_name="../step3_1_results/normalized/Bottle/D00020_refined_simplified_norm.obj",
    csv_path="database/all_features.csv",
    normalization_params_path="database/normalization_params.csv",
    K=10
)

print("\nResults for data/AircraftBuoyant/m1347.obj:")
print("=" * 70)
for i, ((cls, fname), dist) in enumerate(results, 1):
    print(f"{i:2}. {cls:15} | {fname:30} | distance: {dist:8.4f}")

print(f"\nTop-10 classes: {classes}")