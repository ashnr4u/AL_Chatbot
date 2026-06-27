from .pipeline import process_pdf
from .retriever import retrieve
from .generator import generate_response
from .evaluator import evaluate_answer
from .evaluation_data import EVALUATION_DATA


def main():

    embedding_model, chunks, index = process_pdf()

    overall_scores = []

    print("=" * 70)
    print("RAG Evaluation")
    print("=" * 70)

    for i, sample in enumerate(EVALUATION_DATA, start=1):

        question = sample["question"]

        ground_truth = sample["ground_truth"]

        retrieved = retrieve(
            query=question,
            embedding_model=embedding_model,
            index=index,
            chunks=chunks,
        )
        print("\nRetrieved Chunks:", len(retrieved))

        for chunk in retrieved:
            print("-" * 80)
            print("Score:", chunk["score"])
            print(chunk["text"][:300])

            context = "\n\n".join(
                chunk["text"] for chunk in retrieved
            )

        answer = generate_response(
            question,
            context,
        )

        evaluation = evaluate_answer(
            question,
            context,
            answer,
            ground_truth,
        )

        overall_scores.append(evaluation["overall"])

        print("\n")
        print("=" * 70)
        print(f"Question {i}")
        print("=" * 70)

        print("Question:")
        print(question)

        print("\nGenerated Answer:")
        print(answer)

        print("\nEvaluation")

        print(f"Groundedness : {evaluation['groundedness']}/10")
        print(f"Relevance    : {evaluation['relevance']}/10")
        print(f"Completeness : {evaluation['completeness']}/10")
        print(f"Correctness  : {evaluation['correctness']}/10")
        print(f"Overall      : {evaluation['overall']}/10")

        print("\nFeedback:")
        print(evaluation["feedback"])

    print("\n")
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(f"Average Score : {sum(overall_scores)/len(overall_scores):.2f}/10")


if __name__ == "__main__":
    main()