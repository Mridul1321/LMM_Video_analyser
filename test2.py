def check_inclusion(expected, generated):
    """
    Check if all expected keywords are present in the generated output.
    """
    expected_keywords = expected.split(',')
    return all(keyword.lower().strip() in generated.lower() for keyword in expected_keywords)


def analyze_benchmark(data):
    results = []
    total_questions = len(data)
    correct_count = 0

    for entry in data:
        question = entry["question"]
        expected = entry["expected_output"]
        generated = entry["generated_output"]

        is_correct = check_inclusion(expected, generated)
        if is_correct:
            correct_count += 1

        results.append({
            "Question": question,
            "Expected Output": expected,
            "Generated Output": generated,
            "Status": "Correct" if is_correct else "Incorrect"
        })

    accuracy = (correct_count / total_questions) * 100

    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%\n")

    for result in results:
        print(f"Question: {result['Question']}")
        print(f"Expected Output: {result['Expected Output']}")
        print(f"Generated Output: {result['Generated Output']}")
        print(f"Status: {result['Status']}\n")


# Example benchmark data
benchmark_data = [
    {
        "question": "What year did India gain independence?",
        "expected_output": "1947",
        "generated_output": "India gained independence in 1947."
    },
    {
        "question": "Which two countries were created after India's partition?",
        "expected_output": "India,Pakistan",
        "generated_output": "The countries formed were India and Pakistan."
    },
    {
        "question": "Who was the first Prime Minister of India?",
        "expected_output": "Jawaharlal Nehru",
        "generated_output": "The first Prime Minister of India was Nehru."
    }
]
import json
with open(r"E:\Video analysic final year project\audio text to llm\LMM_Video_analyser\model_2.json", "r") as file:
    data = json.load(file)

analyze_benchmark(data)