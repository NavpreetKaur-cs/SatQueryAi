#!/usr/bin/env python
"""Comprehensive CDVQA Evaluation - All Splits, Question Types, and Edge Cases."""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from modules.change_analysis.interface import change_analysis_tool

def categorize_question(query: str) -> str:
    """Categorize question by type."""
    query_lower = query.lower()
    
    if re.search(r"(did|have|did.*change)", query_lower):
        return "binary_change"
    elif re.search(r"(increas|decreas)", query_lower):
        return "directional_change"
    elif re.search(r"(what|where|how|describe)", query_lower):
        return "descriptive"
    elif re.search(r"(vegetat|tree|grass|green)", query_lower):
        return "vegetation_specific"
    elif re.search(r"(water|lake|river|flood)", query_lower):
        return "water_specific"
    elif re.search(r"(built|building|urban|house|road)", query_lower):
        return "buildup_specific"
    else:
        return "other"


def detect_edge_case(query: str, change_pct: float, confidence: float) -> List[str]:
    """Detect potential edge cases."""
    cases = []
    
    if change_pct < 1.0:
        cases.append("very_small_change")
    if change_pct > 50.0:
        cases.append("large_widespread_change")
    if confidence < 0.6:
        cases.append("low_confidence")
    if confidence > 0.95:
        cases.append("very_high_confidence")
    if len(query) > 100:
        cases.append("long_query")
    
    return cases or ["normal"]


def answer_matches_reference(prediction: str, reference: str, confidence: float) -> bool:
    """Match binary CDVQA answers without treating negated change as positive."""

    if reference not in ("yes", "no"):
        return any(
            word in prediction
            for word in reference.split()
            if len(word) > 3
        )

    normalized = prediction.lower()
    negative_patterns = (
        "no material change",
        "no significant change",
        "did not materially change",
        "did not change",
        "no change detected",
        "not changed",
        "unchanged",
        "did not increase",
        "did not decrease",
    )
    is_negative = any(pattern in normalized for pattern in negative_patterns)
    has_positive_change = any(
        word in normalized
        for word in ("increased", "decreased", "changed", "loss", "gain", "affect")
    ) and not is_negative

    if reference == "yes":
        return has_positive_change and confidence > 0.65

    return is_negative or (not has_positive_change and confidence < 0.75)


def evaluate_split(
    split_name: str,
    images_path: Path,
    questions_path: Path,
    answers_path: Path,
    second_dir: Path,
    max_images: int = 100,
    max_questions_per_image: int = 3,
) -> Dict[str, Any]:
    """Comprehensive evaluation on a dataset split."""
    
    print(f"\n{'='*70}")
    print(f"EVALUATING {split_name.upper()} SPLIT")
    print(f"{'='*70}")
    
    try:
        # Load data
        with open(images_path) as f:
            images_data = json.load(f)["images"]
        with open(questions_path) as f:
            questions_data = json.load(f)["questions"]
        with open(answers_path) as f:
            answers_data = json.load(f)["answers"]
        
        # Create lookups
        questions_by_id = {q["id"]: q for q in questions_data}
        answers_by_qid = {a["question_id"]: a for a in answers_data}
        
        # Metrics
        results_by_type = defaultdict(lambda: {"correct": 0, "total": 0, "confidence_sum": 0.0})
        results_by_edge = defaultdict(lambda: {"correct": 0, "total": 0})
        all_results = []
        
        seen_files = set()
        processed_images = 0
        processed_questions = 0
        
        print(f"Processing up to {max_images} images with up to {max_questions_per_image} questions each...")
        
        for img_entry in images_data:
            if processed_images >= max_images:
                break
            
            file_name = img_entry["file_name"]
            if file_name in seen_files:
                continue
            
            seen_files.add(file_name)
            question_ids = img_entry.get("questions_ids", [])[:max_questions_per_image]
            
            if not question_ids:
                continue
            
            # Image paths
            before_path = second_dir / "im1" / file_name
            after_path = second_dir / "im2" / file_name
            
            if not (before_path.exists() and after_path.exists()):
                continue
            
            # Process questions for this image
            for q_id in question_ids:
                question_obj = questions_by_id.get(q_id)
                answer_obj = answers_by_qid.get(q_id)
                
                if not (question_obj and answer_obj):
                    continue
                
                query = question_obj.get("question", "")
                ref_answer = answer_obj.get("answer", "")
                
                if not (query and ref_answer):
                    continue
                
                try:
                    result = change_analysis_tool(
                        str(before_path),
                        str(after_path),
                        query,
                        {"output_dir": "outputs/evaluation", "save_visuals": False}
                    )
                    
                    if result["success"]:
                        pred = result["answer"].lower()
                        ref = ref_answer.lower()
                        conf = result["confidence"]
                        change_pct = result["metadata"].get("change_percentage", 0)
                        
                        # Categorize question
                        q_type = categorize_question(query)
                        
                        # Match prediction to reference.
                        match = answer_matches_reference(pred, ref, conf)
                        
                        # Detect edge cases
                        edge_cases = detect_edge_case(query, change_pct, conf)
                        
                        # Update metrics by type
                        results_by_type[q_type]["total"] += 1
                        results_by_type[q_type]["confidence_sum"] += conf
                        if match:
                            results_by_type[q_type]["correct"] += 1
                        
                        # Update metrics by edge case
                        for edge in edge_cases:
                            results_by_edge[edge]["total"] += 1
                            if match:
                                results_by_edge[edge]["correct"] += 1
                        
                        # Store result
                        all_results.append({
                            "image": file_name,
                            "query": query,
                            "reference": ref_answer,
                            "prediction": result["answer"],
                            "match": match,
                            "confidence": conf,
                            "question_type": q_type,
                            "edge_cases": edge_cases,
                            "change_pct": change_pct,
                        })
                        
                        processed_questions += 1
                        
                except Exception as e:
                    pass
            
            processed_images += 1
            if processed_images % 10 == 0:
                print(f"  Processed {processed_images} images, {processed_questions} questions...")
        
        # Calculate overall accuracy
        total_correct = sum(r["correct"] for r in results_by_type.values())
        total_questions = sum(r["total"] for r in results_by_type.values())
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        
        # Print results by question type
        print(f"\n--- BY QUESTION TYPE ---")
        for q_type in sorted(results_by_type.keys()):
            stats = results_by_type[q_type]
            acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            avg_conf = (stats["confidence_sum"] / stats["total"]) if stats["total"] > 0 else 0
            print(f"{q_type:25s}: {acc:6.2f}% ({stats['correct']:3d}/{stats['total']:3d}), "
                  f"Conf: {avg_conf:.4f}")
        
        # Print results by edge case
        print(f"\n--- BY EDGE CASE ---")
        for edge_type in sorted(results_by_edge.keys()):
            stats = results_by_edge[edge_type]
            acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"{edge_type:25s}: {acc:6.2f}% ({stats['correct']:3d}/{stats['total']:3d})")
        
        # Overall summary
        print(f"\n--- OVERALL RESULTS ---")
        print(f"Images Processed:  {processed_images}")
        print(f"Questions Tested:  {processed_questions}")
        print(f"Accuracy:          {overall_accuracy:.2f}%")
        print(f"Correct:           {total_correct}/{total_questions}")
        
        return {
            "split": split_name,
            "images_processed": processed_images,
            "questions_tested": processed_questions,
            "overall_accuracy": round(overall_accuracy, 2),
            "by_type": {
                q_type: {
                    "accuracy": round(stats["correct"]/stats["total"]*100 if stats["total"] > 0 else 0, 2),
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "avg_confidence": round(stats["confidence_sum"]/stats["total"] if stats["total"] > 0 else 0, 4),
                }
                for q_type, stats in results_by_type.items()
            },
            "by_edge_case": {
                edge: {
                    "accuracy": round(stats["correct"]/stats["total"]*100 if stats["total"] > 0 else 0, 2),
                    "correct": stats["correct"],
                    "total": stats["total"],
                }
                for edge, stats in results_by_edge.items()
            },
            "sample_results": all_results[:10],  # First 10 for inspection
        }
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"split": split_name, "error": str(e)}


def main():
    """Run comprehensive evaluation on all splits."""
    
    cdvqa_dir = Path("data/cdvqa")
    second_dir = Path(r"C:\second dataset\SECOND_train_set")
    
    print("="*70)
    print("COMPREHENSIVE CDVQA EVALUATION")
    print("All Splits, Question Types, Multiple Questions, and Edge Cases")
    print("="*70)
    
    all_splits_results = []
    
    # Evaluate each split
    for split in ["Train", "Val", "Test"]:
        images_path = cdvqa_dir / f"{split}_images.json"
        questions_path = cdvqa_dir / f"{split}_questions.json"
        answers_path = cdvqa_dir / f"{split}_answers.json"
        
        if images_path.exists() and questions_path.exists() and answers_path.exists():
            result = evaluate_split(
                split,
                images_path,
                questions_path,
                answers_path,
                second_dir,
                max_images=100,
                max_questions_per_image=3,
            )
            all_splits_results.append(result)
    
    # Final summary
    print(f"\n{'='*70}")
    print("COMPREHENSIVE SUMMARY")
    print(f"{'='*70}")
    
    for result in all_splits_results:
        if "error" not in result:
            print(f"\n{result['split']} Split:")
            print(f"  Overall Accuracy: {result['overall_accuracy']:.2f}%")
            print(f"  Images: {result['images_processed']}, Questions: {result['questions_tested']}")
    
    # Save detailed results
    results_file = Path("outputs/evaluation/cdvqa_comprehensive_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, "w") as f:
        json.dump({
            "evaluation_type": "Comprehensive CDVQA Evaluation",
            "splits": all_splits_results,
            "timestamp": "2026-09-12",
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {results_file}")


if __name__ == "__main__":
    main()
