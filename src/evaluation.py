from Levenshtein import distance as levenshtein_distance


def evaluate_results(results):
    total = len(results)
    correct = 0
    differences = []

    for filename, (predicted, actual) in results.items():
        if predicted == actual:
            correct += 1
        else:
            diff = {
                "filename": filename,
                "predicted": predicted,
                "actual": actual,
                "distance": levenshtein_distance(predicted, actual),
            }
            differences.append(diff)

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy, differences
