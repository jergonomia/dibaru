from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

TRAINING_DATA = Path("../manual_stance/stance_training_data.csv")

def load_annotated_rag(rag_path, annotation_path, method_name):
    rag = pd.read_csv(rag_path, dtype=str)
    ann = pd.read_csv(annotation_path, dtype=str)

    merged = rag.merge(
        ann,
        on=["query_idx", "run_idx"],
        how="inner",
        suffixes=("", "_manual"),
    )

    merged["stance"] = (
        merged["stance"]
        .str.upper()
        .replace({"UNK": "NEU"})
    )

    merged["method"] = method_name

    merged["classifier_input"] = (
        "Topic: " + merged["original_query"].fillna("") +
        "\nRephrased query: " + merged["rephrased_query"].fillna("") +
        "\nAnswer: " + merged["answer"].fillna("")
    )

    return merged[
        [
            "method",
            "query_idx",
            "run_idx",
            "original_query",
            "rephrased_query",
            "answer",
            "stance",
            "classifier_input",
        ]
    ]

if TRAINING_DATA.exists():
    print("Loading existing training data...")
    df = pd.read_csv(TRAINING_DATA, dtype=str)
else:
    datasets = [
        load_annotated_rag(
            "../manual_stance/Authchain/rag_answers_clean_con_nomic.csv",
            "../manual_stance/Authchain/clean_con_nomic_annotation.csv",
            "Authchain_clean",
        ),
        load_annotated_rag(
            "../manual_stance/Authchain/rag_answers_poison_con_nomic.csv",
            "../manual_stance/Authchain/poison_con_nomic_annotation.csv",
            "Authchain_poison",
        ),
        load_annotated_rag(
            "../manual_stance/topic+new_method/rag_answers_poison_con_nomic.csv",
            "../manual_stance/topic+new_method/poison_con_nomic_annotation.csv",
            "topic_new_method_poison",
        ),
    ]


    df = pd.concat(datasets, ignore_index=True)

    print(df["method"].value_counts())
    print(df["stance"].value_counts())

    df.to_csv("../manual_stance/stance_training_data.csv", index=False)


def main():
    X = df["classifier_input"]
    y = df["stance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    X_train_emb = embedder.encode(
        X_train.tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    X_test_emb = embedder.encode(
        X_test.tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    clf.fit(X_train_emb, y_train)

    pred = clf.predict(X_test_emb)
    print(classification_report(y_test, pred))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        display_labels=["PRO", "CON", "NEU"],
        cmap="Blues",
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()