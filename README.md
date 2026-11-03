Age-Appropriateness Classification of Russian Children's Fiction

Code for my master's thesis, Building a Linguistic Corpus for Generating Personalised Fairy Tales for Children Aged 5–12. I defended it at Tomsk State University in 2026 and graduated with honours.

Task: predict whether a Russian children's text suits preschool children (ages 5–6) or primary-school children (ages 7–12). The goal is a classifier that can check the age suitability of text produced by story-generation systems.

Best result: a fine-tuned cointegrated/rubert-tiny2 model reaches a document-level Macro F1 of 0.767. The best CNN, using static fastText embeddings, reaches 0.742.

Corpus

The corpus is built from RusAge, a corpus of Russian fiction labelled with age ratings, downloaded through the Kaggle API.

I removed texts rated 3+, 16+ and 18+ and merged the remaining ratings into two classes:
1_дошкольная_группа (ages 5–6): 1,243 texts
2_младшая_школа (ages 7–12): 1,559 texts
That gives 2,802 documents in total. Their metadata is in full_metadata.csv: file id, title, author, age rating, genre and age group.
Cleaning steps:
fixed issues in the original metadata;
removed LitRes service blocks, publisher notes, front matter and empty lines;
merged the texts and metadata into one JSONL file.

The raw texts are not included in this repository because of their size and licensing.

Method
Sentence-aware chunking (scripts/chunker.py). Long documents are split into chunks that stay within the model's context limit without breaking sentences. Each chunk keeps a doc_id and a chunk_id.
Document-level split. Train, validation and test sets are split by document rather than by chunk, so no text leaks between them.
Models
CNN (scripts/CNN_Scripts/): I tried a baseline TextCNN with embeddings trained from scratch, static fastText embeddings, and multichannel variants. All use class-weighted cross-entropy to handle the class imbalance.
BERT (scripts/BERT_Scripts/): cointegrated/rubert-tiny2, fine-tuned with different chunk lengths, learning rates, and full or partial fine-tuning.
Document-level aggregation. Predictions for the chunks of each document are combined using one of four methods: probability averaging, majority vote, weighted vote or temperature scaling.
Results
Model	Configuration	Aggregation	Document-level Macro F1
BERT	rubert-tiny2, chunk length 510, full fine-tuning, lr 3e-5	best of the four methods	0.767
CNN	fastText (static)	majority vote	0.742

The two architectures score close to each other. Whether a text suits a given age depends on more than vocabulary: plot structure, theme and narrative complexity matter too. This is why judging the whole document works better than judging individual chunks.

Repository structure
scripts/
├── download_rusage.py         # download the RusAge corpus
├── eda_rusage.py              # exploratory data analysis
├── copy_selected_texts.py     # pick texts for the target age range
├── remove_frontmatter.py      # remove service and publisher text
├── json_creation.py           # merge texts and metadata into JSONL
├── chunker.py                 # sentence-aware chunking
├── CNN_Scripts/               # CNN models, training and document-level evaluation
└── BERT_Scripts/              # BERT model, training and document-level evaluation
full_metadata.csv              # metadata for the 2,802 documents
Stack

Python, PyTorch, Hugging Face Transformers, scikit-learn, fastText, pandas, NumPy.

Author

Mika Swagemakers. The full thesis is in Russian; I'm happy to walk through it in English.
