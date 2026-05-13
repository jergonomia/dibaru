import csv
import ollama



def main():
    client = ollama.Client()

    model = "gemma3:4b"
    prompt_path = "data/queries.docdev.tsv"
    out_path = "out.csv"

    max_rows = 5
    with open(prompt_path, "r", encoding="utf-8") as f, \
        open(out_path, "w", encoding="utf-8") as out:

        reader = csv.reader(f, delimiter="\t")
        writer = csv.writer(out)

        for i, row in enumerate(reader):

            if i == max_rows:
                break

            prompt = row[1].strip()

            print(f"Running prompt {i+1}: {prompt}")

            try:
                response = client.generate(model=model, prompt=prompt)
            except Exception as e:
                response = f"Error: {e}"

            writer.writerow([prompt, response.response])


    print(f"Finished, model generated outputs at {out_path}")



main()


