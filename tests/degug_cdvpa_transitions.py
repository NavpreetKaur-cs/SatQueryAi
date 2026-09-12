from modules.change_analysis.semantic_labels import analyze_second_labels


LABEL1_DIR = r"C:\second dataset\SECOND_train_set\label1"
LABEL2_DIR = r"C:\second dataset\SECOND_train_set\label2"


images = [
    "10589.png",
    "06340.png",
    "00328.png",
    "01889.png",
]


for filename in images:

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    result = analyze_second_labels(
        f"{LABEL1_DIR}\\{filename}",
        f"{LABEL2_DIR}\\{filename}",
    )

    print(f"\nChanged:   {result['changed_percentage']:.2f}%")
    print(f"Unchanged: {result['unchanged_percentage']:.2f}%")

    print("\nBefore percentages:")
    for name, value in result["before_percent"].items():
        print(f"  {name:20s}: {value:.2f}%")

    print("\nAfter percentages:")
    for name, value in result["after_percent"].items():
        print(f"  {name:20s}: {value:.2f}%")

    print("\nTransitions:")

    for source, destinations in result["transitions"].items():

        if not destinations:
            continue

        print(f"\n  {source}:")

        for destination, pixels in destinations.items():
            print(
                f"    -> {destination:20s}: {pixels:6d} pixels"
            )