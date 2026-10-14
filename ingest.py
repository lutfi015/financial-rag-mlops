import json

# Sample data finansial dummy dengan struktur persis 'financial_phrasebank'
BASE_SAMPLES = [
    (
        "According to Gran, the company has no plans to move production to Russia.",
        "neutral",
    ),
    (
        "Technopolis plans to develop in stages an area of no less than 100,000 square meters.",
        "neutral",
    ),
    (
        "The international electronic industry company Elcoteq has laid off tens of employees.",
        "negative",
    ),
    (
        "With the new production plant company expected to increase its capacity by 50%.",
        "positive",
    ),
    (
        "Operating profit rose to EUR 12.5 million from EUR 8.7 million in the corresponding period.",
        "positive",
    ),
    (
        "Net sales decreased by 10% to EUR 42.1 million due to low demand in European market.",
        "negative",
    ),
    (
        "The board of directors proposes a dividend of EUR 0.50 per share for the fiscal year.",
        "positive",
    ),
    (
        "Shares of the company were down 3.2% in early morning trading on the Helsinki exchange.",
        "negative",
    ),
]


def prepare_financial_data(num_samples: int = 200):
    print(" Generating financial dataset locally...")

    documents = []
    for idx in range(num_samples):
        base_text, sentiment = BASE_SAMPLES[idx % len(BASE_SAMPLES)]
        documents.append(
            {
                "id": idx,
                "text": f"{base_text} (Record #{idx+1})",
                "sentiment": sentiment,
            }
        )

    with open("financial_docs.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(
        f" Success! {len(documents)} documents saved to 'financial_docs.json'."
    )


if __name__ == "__main__":
    prepare_financial_data(num_samples=200)