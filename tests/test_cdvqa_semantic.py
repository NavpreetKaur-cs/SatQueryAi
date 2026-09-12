import json
import sys
from pathlib import Path


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)


from modules.change_analysis.semantic_labels import (
    analyze_second_labels,
    answer_semantic_question,
    normalize_semantic_answer,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


CDVQA_DIR = (
    PROJECT_ROOT
    / "data"
    / "cdvqa"
)


SECOND_DIR = Path(
    r"C:\second dataset\SECOND_train_set"
)


# ============================================================
# SETTINGS
# ============================================================

# Number of UNIQUE image pairs to test
MAX_IMAGES = 100


# ============================================================
# LOAD JSON
# ============================================================

def load_json(filename):

    path = CDVQA_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


images_data = load_json(
    "Train_images.json"
)

questions_data = load_json(
    "Train_questions.json"
)

answers_data = load_json(
    "Train_answers.json"
)


# ============================================================
# LOAD DATA
# ============================================================

image_entries = images_data["images"]

questions_list = questions_data["questions"]

answers_list = answers_data["answers"]


# ============================================================
# GROUP QUESTIONS BY UNIQUE IMAGE
# ============================================================

image_question_map = {}


for entry in image_entries:

    image_id = entry.get(
        "file_name"
    )

    if not image_id:
        continue


    question_ids = entry.get(
        "questions_ids",
        []
    )


    if image_id not in image_question_map:

        image_question_map[
            image_id
        ] = []


    image_question_map[
        image_id
    ].extend(question_ids)


# ============================================================
# REMOVE DUPLICATE QUESTION IDs
# ============================================================

for image_id in image_question_map:

    image_question_map[
        image_id
    ] = list(
        dict.fromkeys(
            image_question_map[
                image_id
            ]
        )
    )


# ============================================================
# FIND VALID UNIQUE IMAGE PAIRS
# ============================================================

valid_images = []


for image_id, question_ids in (
    image_question_map.items()
):

    label1 = (
        SECOND_DIR
        / "label1"
        / image_id
    )

    label2 = (
        SECOND_DIR
        / "label2"
        / image_id
    )


    if (
        label1.exists()
        and label2.exists()
        and len(question_ids) > 0
    ):

        valid_images.append({
            "image_id": image_id,
            "question_ids": question_ids,
        })


# ============================================================
# LIMIT TEST
# ============================================================

images_to_test = (
    valid_images[:MAX_IMAGES]
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("=" * 70)
print(
    "CDVQA MULTI-IMAGE SEMANTIC EVALUATION"
)
print("=" * 70)

print(
    f"Unique image entries found: "
    f"{len(image_question_map)}"
)

print(
    f"Valid image pairs found: "
    f"{len(valid_images)}"
)

print(
    f"Unique image pairs selected: "
    f"{len(images_to_test)}"
)

print("=" * 70)


# ============================================================
# GLOBAL RESULTS
# ============================================================

total_questions = 0
total_correct = 0

total_images = 0
perfect_images = 0

failed_questions = []


# ============================================================
# PROCESS IMAGES
# ============================================================

for index, item in enumerate(
    images_to_test,
    start=1
):

    image_id = item[
        "image_id"
    ]

    question_ids = item[
        "question_ids"
    ]


    label1 = (
        SECOND_DIR
        / "label1"
        / image_id
    )

    label2 = (
        SECOND_DIR
        / "label2"
        / image_id
    )


    print()
    print("=" * 70)

    print(
        f"IMAGE {index}/"
        f"{len(images_to_test)}: "
        f"{image_id}"
    )

    print("=" * 70)

    print(
        f"Questions: "
        f"{len(question_ids)}"
    )

    print(
        f"Label 1: "
        f"{label1.exists()}"
    )

    print(
        f"Label 2: "
        f"{label2.exists()}"
    )


    # ========================================================
    # ANALYZE LABELS
    # ========================================================

    try:

        analysis = (
            analyze_second_labels(
                label1,
                label2
            )
        )

    except Exception as error:

        print()
        print(
            "ERROR during semantic analysis:"
        )

        print(error)

        continue


    print()

    print(
        f"Changed area: "
        f"{analysis['changed_percentage']:.2f}%"
    )

    print(
        f"Unchanged area: "
        f"{analysis['unchanged_percentage']:.2f}%"
    )


    # ========================================================
    # EVALUATE QUESTIONS
    # ========================================================

    image_correct = 0
    image_total = 0


    for question_id in question_ids:

        # Safety checks
        if (
            question_id < 0
            or
            question_id >= len(
                questions_list
            )
        ):

            continue


        if (
            question_id < 0
            or
            question_id >= len(
                answers_list
            )
        ):

            continue


        question_entry = (
            questions_list[
                question_id
            ]
        )

        answer_entry = (
            answers_list[
                question_id
            ]
        )


        question = (
            question_entry[
                "question"
            ]
        )

        ground_truth = (
            answer_entry[
                "answer"
            ]
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        predicted = (
            answer_semantic_question(
                question,
                analysis
            )
        )


        # ----------------------------------------------------
        # COMPARE
        # ----------------------------------------------------

        is_correct = (
    normalize_semantic_answer(predicted)
    ==
    normalize_semantic_answer(ground_truth)
)

        image_total += 1
        total_questions += 1


        if is_correct:

            image_correct += 1
            total_correct += 1

        else:

            failed_questions.append({

                "image":
                    image_id,

                "question_id":
                    question_id,

                "question":
                    question,

                "ground_truth":
                    ground_truth,

                "predicted":
                    predicted,

            })


    # ========================================================
    # IMAGE RESULT
    # ========================================================

    if image_total > 0:

        image_accuracy = (
            image_correct
            /
            image_total
            *
            100
        )

    else:

        image_accuracy = 0


    print()

    print(
        f"Image result: "
        f"{image_correct}/"
        f"{image_total}"
    )

    print(
        f"Image accuracy: "
        f"{image_accuracy:.2f}%"
    )


    total_images += 1


    if (
        image_correct
        ==
        image_total
    ):

        perfect_images += 1


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print()
print("=" * 70)
print(
    "FINAL MULTI-IMAGE RESULTS"
)
print("=" * 70)

print(
    f"Images tested: "
    f"{total_images}"
)

print(
    f"Images with 100% accuracy: "
    f"{perfect_images}"
)

print(
    f"Total questions: "
    f"{total_questions}"
)

print(
    f"Correct answers: "
    f"{total_correct}"
)

print(
    f"Incorrect answers: "
    f"{total_questions - total_correct}"
)


# ============================================================
# OVERALL ACCURACY
# ============================================================

if total_questions > 0:

    overall_accuracy = (
        total_correct
        /
        total_questions
        *
        100
    )

else:

    overall_accuracy = 0


print(
    f"Overall accuracy: "
    f"{overall_accuracy:.2f}%"
)


# ============================================================
# FAILED QUESTIONS
# ============================================================

print()
print("=" * 70)
print(
    "FAILED QUESTIONS"
)
print("=" * 70)


if not failed_questions:

    print(
        "No failed questions!"
    )

else:

    for failure in failed_questions:

        print()

        print(
            f"Image: "
            f"{failure['image']}"
        )

        print(
            f"Question ID: "
            f"{failure['question_id']}"
        )

        print(
            f"Question: "
            f"{failure['question']}"
        )

        print(
            f"Ground truth: "
            f"{failure['ground_truth']}"
        )

        print(
            f"Predicted: "
            f"{failure['predicted']}"
        )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)


if (
    total_questions > 0
    and
    total_correct == total_questions
):

    print(
        "ALL TESTED CDVQA QUESTIONS PASSED!"
    )

elif total_questions > 0:

    print(
        "Some CDVQA questions failed."
    )

else:

    print(
        "No questions were evaluated."
    )


print("=" * 70)