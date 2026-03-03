# main.py
import argparse
from src.pipeline import rank, rerank_pipeline

DATA_PATH = "data/potential-talents.csv"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH, help="Path to the dataset CSV file")
    parser.add_argument("--star_id", type=int, default=None, help="ID of the star talent to rerank around")
    parser.add_argument("--output", default="data/ranked-talents.csv", help="Path to the output CSV file")

    args = parser.parse_args()

    df = rank(args.data)

    if args.star_id is None:
        df.to_csv(args.output, index=False)
        print("Baseline ranking saved to:", args.output)
        return
    
    # Adaptive reranking
    df = rerank_pipeline(df, args.star_id)
    df.to_csv(args.output, index=False)
    print("Adaptive ranking saved to:", args.output)

    print("\nTop 20 for star_id:", args.star_id)
    print(df.head(20))

if __name__ == "__main__":
    main()