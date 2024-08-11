# Abstract_Decoder
Text Classification Model for Medical Abstracts

Built and deployed a CNN model to classify sentences in medical abstracts with 83.5% accuracy  in distinguishing between different sections of abstracts (e.g., Background, Methods, Results) using TensorFlow.
Developed preprocessing pipelines for text vectorization and embedding, enhancing model performance on large-scale medical datasets.
Utilized pretrained embeddings (e.g., Universal Sentence Encoder) to improve feature extraction and classification accuracy.

### 1. **Environment Setup and Data Loading**

- **GPU Check:** Confirm you have access to GPU with `!nvidia-smi -L`.
- **Data Download:** Clone the dataset from GitHub.
  ```python
  !git clone https://github.com/Franck-Dernoncourt/pubmed-rct
  ```

### 2. **Data Inspection and Preprocessing**

- **List Files:** Verify dataset files.
  ```python
  !ls pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign/
  ```

- **Preprocessing Function:**
  The `preprocess_text_with_line_numbers` function formats your data into dictionaries. This approach helps with creating a structured dataset.

- **Visualization:** You visualized the data by converting it into dataframes and plotting distributions, which is excellent for understanding the data characteristics.

### 3. **Vectorization and Embedding**

- **Text Vectorization:** Convert text to numerical format using `TextVectorization` and `Embedding` layers.
  ```python
  from tensorflow.keras.layers import TextVectorization, Embedding
  ```

- **Universal Sentence Encoder:** Use pre-trained embeddings from TensorFlow Hub for feature extraction.
  ```python
  import tensorflow_hub as hub
  ```

### 4. **Modeling**

- **Baseline Model:** You started with a TF-IDF + Naive Bayes model as a baseline.
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.naive_bayes import MultinomialNB
  ```

- **Conv1D Model:** Implemented a Conv1D model for sequence processing with token embeddings.
  ```python
  from tensorflow.keras import layers
  ```

- **Feature Extraction with Pretrained Embeddings:** Used TensorFlow Hub's Universal Sentence Encoder.
  ```python
  tf_hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False)
  ```

- **Character-Level Embeddings:** Created character-level tokenization and vectorization for fine-grained text representation.
  ```python
  import string
  char_vectorizer = TextVectorization(max_tokens=NUM_CHAR_TOKENS, output_sequence_length=output_seq_char_len)
  ```

### Tips and Next Steps:

1. **Evaluate Models:** Continuously compare your models' performance using metrics and validation results. It’s crucial to track metrics like accuracy, precision, recall, and F1-score.

2. **Experiment with Hyperparameters:** Tune hyperparameters for better model performance. For example, you could adjust the number of filters in Conv1D layers, sequence length, or embedding dimensions.

3. **Explore Advanced Models:** Consider exploring more advanced models like BERT or GPT if the simpler models don’t meet your needs. These models can capture more complex relationships in text data.

4. **Ensure Data Balance:** Check for class imbalances and use techniques such as oversampling or undersampling if necessary to improve model performance.

5. **Consider Deployment:** Once you have a robust model, think about how to deploy it for real-world use. This might involve creating a web service or integrating it into an application.

Let me know if you need any more help with specific parts of the process or further clarification!
