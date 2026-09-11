from modules.change_analysis.interface import change_analysis_tool


result = change_analysis_tool(
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

print("\n=== INTERFACE TEST ===")
print("Success:", result["success"])
print("Answer:", result["answer"])
print("Confidence:", result["confidence"])

print("\nVisual outputs:")

for name, path in result["metadata"]["visuals"].items():
    print(f"{name}: {path}")