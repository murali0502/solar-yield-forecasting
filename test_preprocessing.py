from pathlib import Path

from src.preprocessing.data_loader import (
    prepare_dataset,
    chronological_split,
)


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "solar_dataset.csv"
)


def main():

    df = prepare_dataset(DATA_PATH)

    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(df.head().to_string())

    print()
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    train, validation, test = chronological_split(df)

    print()
    print(
        "Preprocessing test completed successfully."
    )


if __name__ == "__main__":
    main()