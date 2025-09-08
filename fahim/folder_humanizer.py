import os
from transformers import pipeline

# Load paraphraser once (so it doesn't reload each time)
paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

def humanize_text(text: str) -> str:
    """Paraphrase and humanize input text."""
    result = paraphraser(
        text,
        max_length=10000,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    return result[0]['generated_text']

def process_folder(folder_path: str):
    """Finds all .txt files in folder, paraphrases them, and saves output."""
    if not os.path.isdir(folder_path):
        print("❌ Invalid folder path.")
        return

    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    if not txt_files:
        print("⚠️ No .txt files found in the folder.")
        return

    print(f"\n📂 Found {len(txt_files)} text file(s). Processing...\n")

    for file in txt_files:
        input_path = os.path.join(folder_path, file)
        output_path = os.path.join(folder_path, f"humanized_{file}")

        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"➡️ Humanizing: {file}")

        humanized = humanize_text(content)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(humanized)

        print(f"✅ Saved: {output_path}\n")

if __name__ == "__main__":   # <-- double underscores before/after
    while True:
        folder = input("\n📁 Enter folder path containing .txt files (or type 'exit' to quit): ").strip()

        if folder.lower() in ["exit", "quit"]:
            print("👋 Exiting program. Goodbye!")
            break

        process_folder(folder)
