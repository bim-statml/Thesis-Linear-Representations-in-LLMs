# Linear Representations
In our report, we explore how the notion of a linear representation hypothesis applies to multilingual concepts in large language models

We confirm the results with LLaMA-2 representations and this repo provides the code for the experiments.

## Data
In [`word_pairs`](word_pairs), each `[___ - ___].txt` has bilingual pairs of words for each language combination. We use them to estimate the unembedding representations.

In [`paired_contexts`](paired_contexts), each `__-__.jsonl` has context samples from Wikipedia in different languages. We use them for the measurement experiment ([`3_measurement.ipynb`](3_measurement.ipynb)).

## Requirement
You need to install Python packages `transformers`, `torch`, `numpy`, `seaborn`, `matplotlib`, `json`, and `tqdm` to run the codes. Also, you need some GPUs to implement the code efficiently.

Make a directory `matrices` and run [`0_store_matrices_bm.py`](0_store_matrices_bm.ipynb) first before you run other jupyter notebooks.

## Experiments
- [**`1_directions.ipynb`**](1_directions.ipynb): We compare the projection of differences between bilingual pairs (vs random pairs) onto their corresponding concept direction
- [**`2_causal.ipynb`**](2_causal.ipynb): We compare the orthogonality between the unembedding representations for causally separable multilingual concepts based on the causal inner product
- [**`3_classification.ipynb`**](3_classification.ipynb): We confirm that bilingual pairs that are statistically independent under the causal inner product
still carry information about linguistically similar languages.

- [**`4_translation.ipynb`**](4_translation.ipynb): We confirm that the embedding representation changes the target concept, which can be used for rudimentary translation in a bilingual concept setting.
