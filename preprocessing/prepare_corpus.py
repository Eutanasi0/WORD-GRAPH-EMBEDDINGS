from collections import Counter
from pathlib import Path

from gensim.models.word2vec import Text8Corpus

# Rutas del proyecto
RAW_CORPUS = Path("data/raw/text8.gz")
PROCESSED_DIR = Path("data/processed")

# Parámetros iniciales
VOCAB_SIZE = 10_000
MIN_COUNT = 5


def build_vocabulary(corpus_path):
    """Calcula las frecuencias y selecciona el vocabulario."""

    corpus = Text8Corpus(str(corpus_path))
    frequencies = Counter()

    for words in corpus:
        frequencies.update(words)

    vocabulary = [
        (word, count)
        for word, count in frequencies.most_common()
        if count >= MIN_COUNT
    ][:VOCAB_SIZE]

    return vocabulary


def filter_corpus(corpus_path, vocabulary, output_path):
    """Conserva únicamente las palabras del vocabulario."""

    allowed_words = {word for word, _ in vocabulary}
    corpus = Text8Corpus(str(corpus_path))

    with output_path.open("w", encoding="utf-8") as output:
        for words in corpus:
            filtered_words = [
                word for word in words
                if word in allowed_words
            ]

            if filtered_words:
                output.write(" ".join(filtered_words) + "\n")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_CORPUS.exists():
        raise FileNotFoundError(
            f"No se encontró el corpus: {RAW_CORPUS}"
        )

    print("Contabilizando palabras...")
    vocabulary = build_vocabulary(RAW_CORPUS)

    vocab_path = PROCESSED_DIR / "vocabulary.txt"

    with vocab_path.open("w", encoding="utf-8") as file:
        for word, count in vocabulary:
            file.write(f"{word}\t{count}\n")

    print("Generando corpus filtrado...")

    filtered_path = PROCESSED_DIR / "text8_filtered.txt"

    filter_corpus(
        RAW_CORPUS,
        vocabulary,
        filtered_path
    )

    print("\nPreprocesamiento completado.")
    print(f"Palabras seleccionadas: {len(vocabulary)}")
    print(f"Vocabulario: {vocab_path}")
    print(f"Corpus filtrado: {filtered_path}")


if __name__ == "__main__":
    main()
