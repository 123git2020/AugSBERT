# Augmented SBERT
## Background 
Sentence-Transformer framework provides an easy method to compute dense vector representations for **sentences**, **paragraphs**, and **images**. The models are based on transformer networks like BERT / RoBERTa / XLM-RoBERTa etc. and achieve state-of-the-art performance in various tasks. Text is embedded in vector space such that similar text are closer and can efficiently be found using cosine similarity.

The original paper of Sentence-BERT (SBERT) is **https://arxiv.org/abs/1908.10084**

The original codebase of Sentence-Transformer is **https://github.com/UKPLab/sentence-transformers**

For the full documentation of how to use Sentence-Transformer library, see **[www.SBERT.net](https://www.sbert.net)**.


## Motivation

Bi-encoders (a.k.a. sentence embeddings models) require substantial training data and fine-tuning over the target task to achieve competitive performances. However, in many scenarios, there is only little training data available.
 
 To solve this practical issue, we release an effective data-augmentation strategy known as <b>Augmented SBERT</b> where we utilize a high performing and slow cross-encoder (BERT) to label a larger set of input pairs to augment the training data for the bi-encoder (SBERT).

For more details, refer to our publication - [Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks](https://arxiv.org/abs/2010.08240) which is a joint effort by Nandan Thakur, Nils Reimers and Johannes Daxenberger of UKP Lab, TU Darmstadt.

Chien Vu also wrote a nice blog article on this technique: [Advance BERT model via transferring knowledge from Cross-Encoders to Bi-Encoders](https://towardsdatascience.com/advance-nlp-model-via-transferring-knowledge-from-cross-encoders-to-bi-encoders-3e0fc564f554)

## Extend to your own datasets

**Scenario 1: Limited or small annotated datasets (few labeled sentence-pairs (1k-3k))**\
If you have specialized datasets in your company or research which are small-sized or contain labeled few sentence-pairs. You can extend the idea of Augmented SBERT (in-domain) strategy by training a cross-encoder over your small gold  dataset and use BM25 sampling to generate combinations not seen earlier. Use the cross-encoder to label these unlabeled pairs to create the silver dataset. Finally train a bi-encoder (i.e. SBERT) over your extended dataset (gold+silver) dataset as shown in [train_sts_indomain_bm25.py](train_sts_indomain_bm25.py).

**Scenario 2: No annotated datasets (Only unlabeled sentence-pairs)**\
If you have specialized datasets in your company or research which only contain unlabeled sentence-pairs. You can extend the idea of Augmented SBERT (domain-transfer) strategy by training a cross-encoder over a source dataset which is annotated (for eg. QQP). Use this cross-encoder to label your specialised unlabeled dataset i.e. target dataset. Finally train a bi-encoder i.e. SBERT over your labeled target dataset as shown in [train_sts_qqp_crossdomain.py](train_sts_qqp_crossdomain.py).


## Methodology 
There are two major scenarios for the Augmented SBERT approach for pairwise-sentence regression or classification tasks. 

## Scenario 1: Limited or small annotated datasets (few labeled sentence-pairs)

We apply the Augmented SBERT (<b>In-domain</b>) strategy, it involves three steps - 

 - Step 1:  Train a cross-encoder (BERT) over the small (gold or annotated) dataset

 - Step 2.1: Create pairs by recombination and reduce the pairs via BM25

 - Step 2.2: Weakly label new pairs with cross-encoder (BERT). These are silver pairs or (silver) dataset

 - Step 3:  Finally, train a bi-encoder (SBERT) on the extended (gold + silver) training dataset

<img src="https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/augsbert-indomain.png" width="400" height="500">

## Scenario 2: No annotated datasets (Only unlabeled sentence-pairs)

We apply the Augmented SBERT (<b>Domain-Transfer</b>) strategy, it involves three steps - 

 - Step 1: Train from scratch a cross-encoder (BERT) over a source dataset, for which we contain annotations

 - Step 2: Use this cross-encoder (BERT) to label your target dataset i.e. unlabeled sentence pairs

 - Step 3: Finally, train a bi-encoder (SBERT) on the labeled target dataset

<img src="https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/augsbert-domain-transfer.png" width="500" height="300">


## Training
 
Training examples for each scenario explained below:

- [train_mrpc_indomain_bm25.py](train_mrpc_indomain_bm25.py)
    - Script initially trains a cross-encoder (BERT) model from scratch for small mrpc dataset.
    - Recombine sentences from our small training dataset and form lots of sentence-pairs.
    - Limit number of combinations with BM25 sampling using [FastBM25](https://github.com/zhusleep/fastbm25).
    - Retrieve top-k sentences given a sentence and label these pairs using the cross-encoder (silver dataset).
    - Train a bi-encoder (SBERT) model on both gold + silver STSb dataset. (Augmented SBERT (In-domain) Strategy).

- [train_sts_indomain_bm25.py](train_sts_indomain_bm25.py)
    - Script initially trains a cross-encoder (BERT) model from scratch for small STS benchmark dataset.
    - Recombine sentences from our small training dataset and form lots of sentence-pairs.
    - Limit number of combinations with BM25 sampling using [FastBM25](https://github.com/zhusleep/fastbm25).
    - Retrieve top-k sentences given a sentence and label these pairs using the cross-encoder (silver dataset).
    - Train a bi-encoder (SBERT) model on both gold + silver STSb dataset. (Augmented SBERT (In-domain) Strategy).

- [train_sts_qqp_crossdomain.py](train_sts_qqp_crossdomain.py)
    - This script initially trains a cross-encoder (BERT) model from scratch for STS benchmark dataset.
    - Label the Quora Questions Pair (QQP) training dataset (Assume no labels present) using the cross-encoder.
    - Train a bi-encoder (SBERT) model on the QQP dataset. (Augmented SBERT (Domain-Transfer) Strategy).

- [train_sts_qqp_indomain.py](train_sts_qqp_indomain.py)
    - This script directly trains a bi-encoder (SBERT) model on the positive examples in QQP train dataset
    - When trained with both positive and negative examples, results can get worse


## Citation
If you use the code for augmented sbert, feel free to cite our publication [Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks](https://arxiv.org/abs/2010.08240):
``` 
@article{thakur-2020-AugSBERT,
    title = "Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks",
    author = "Thakur, Nandan and Reimers, Nils and Daxenberger, Johannes and  Gurevych, Iryna", 
    journal= "arXiv preprint arXiv:2010.08240",
    month = "10",
    year = "2020",
    url = "https://arxiv.org/abs/2010.08240",
}
```
