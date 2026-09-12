from modules.change_analysis.pipeline import analyze_change


result = analyze_change(
    r"C:\second dataset\SECOND_train_set\im1\10589.png",
    r"C:\second dataset\SECOND_train_set\im2\10589.png",
    "Has the area changed?",
    {
        "label1_path": r"C:\second dataset\SECOND_train_set\label1\10589.png",
        "label2_path": r"C:\second dataset\SECOND_train_set\label2\10589.png",
        "output_dir": "outputs/change_analysis",
        "save_visuals": True,
    },
)

print("\n=== SEMANTIC CHANGE MAP TEST ===")
print("Success:", result["success"])
print("Answer:", result["answer"])
print("Model:", result["model"])

print("\nGenerated files:")

for name, path in result["metadata"]["visuals"].items():
    print(f"{name}: {path}")

print("\nSemantic map generated:",
      result["metadata"]["semantic_change_map"])