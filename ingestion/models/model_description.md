# Model Description

## 1. Description:

There are two binary files:

- vocab.bin: Contains all of word in model’s knowledge
- sense_embeddings.bin: Contains information of each word such as word, sense id, embedding

## 2. Structure:

- vocab.bin:
  1. **Header**: num_records (4 byte, int32, little-endian)
  1. **Body**: (each word)
     - word_length (4 byte, int32)
     - word (N byte, UTF-8)
     - word_id (4 byte, int32)

- sense_embeddings.bin:
  - **Header**:
    - num_records (4 byte, int32, LE)
    - embedding_dim (4 byte, int32, LE)
  - **Body**: mỗi dòng gồm:
    - word_length (4 byte, int32)
    - word (N byte, UTF-8)
    - sense_id (4 byte, int32)
    - embedding (dim\*4 byte, float32 [])
