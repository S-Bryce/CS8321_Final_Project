---

language: 
  - en
tags:
- coreference-resolution
license: mit
datasets:
- ontonotes
metrics:
- CoNLL
task_categories:
- coreference-resolution
model-index:
- name: biu-nlp/lingmess-coref
  results:
  - task:
      type: coreference-resolution
      name: coreference-resolution
    dataset:
      name: ontonotes
      type: coreference
    metrics:
    - name: Avg. F1
      type: CoNLL
      value: 81.4

---

## LingMess: Linguistically Informed Multi Expert Scorers for Coreference Resolution

[LingMess](https://arxiv.org/abs/2205.12644) is a linguistically motivated categorization of mention-pairs into 6 types of coreference decisions and learn a dedicated trainable scoring function for each category. This significantly improves the accuracy of the pairwise scorer as well as of the overall coreference performance on the English Ontonotes coreference corpus.

Please check the [official repository](https://github.com/shon-otmazgin/lingmess-coref) for more details and updates.

#### Training on OntoNotes

We present the test results on OntoNotes 5.0 dataset.

| Model                           | Avg. F1 |
|---------------------------------|---------|
| SpanBERT-large + e2e            | 79.6    |
| Longformer-large + s2e          | 80.3    |
| **Longformer-large + LingMess** | 81.4    |


### Citation

If you find LingMess useful for your work, please cite the following paper:

``` latex
@misc{https://doi.org/10.48550/arxiv.2205.12644,
  doi = {10.48550/ARXIV.2205.12644},
  url = {https://arxiv.org/abs/2205.12644},
  author = {Otmazgin, Shon and Cattan, Arie and Goldberg, Yoav},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {LingMess: Linguistically Informed Multi Expert Scorers for Coreference Resolution},
  publisher = {arXiv}, 
  year = {2022}, 
  copyright = {Creative Commons Attribution 4.0 International}
}
```
