"""
The script shows how to train Augmented SBERT (Domain-Transfer/Cross-Domain) strategy for STSb-QQP dataset.
For our example below we consider STSb (source) and QQP (target) datasets respectively.

Methodology:
Three steps are followed for AugSBERT data-augmentation strategy with Domain Trasfer / Cross-Domain - 
1. Cross-Encoder aka BERT is trained over STSb (source) dataset.
2. Cross-Encoder is used to label QQP training (target) dataset (Assume no labels/no annotations are provided).
3. Bi-encoder aka SBERT is trained over the labeled QQP (target) dataset.

Citation: https://arxiv.org/abs/2010.08240

Usage:
python train_sts_qqp_crossdomain.py

OR
python train_sts_qqp_crossdomain.py pretrained_transformer_model_name
"""
from torch.utils.data import DataLoader
from sentence_transformers import models, losses, util, LoggingHandler, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
import logging
import csv
import torch
import math
import os
import random

random.seed(86)
torch.manual_seed(9910)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'bert-base-uncased'
batch_size = 16
num_epochs = 1
max_seq_length = 128
use_cuda = torch.cuda.is_available()

###### Read Datasets ######
sts_dataset_path = 'stsbenchmark/data.tsv'
qqp_dataset_path = 'qqp'

bi_encoder_path = 'qqp/bi-encoder_cross_domain_'+model_name.replace("/", "-")

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])


silver_data = []
silver_scores=[]

with open(os.path.join(qqp_dataset_path, "classification/train_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
                if row['is_duplicate'] == '1':
                    silver_data.append([row['question1'], row['question2']])
                    silver_scores.append(int(row['is_duplicate']))


###########################################################################
#
# Step 1: Train bi-encoder (SBERT) model with QQP dataset - Augmented SBERT
#
###########################################################################

logging.info("Step 1: Train bi-encoder: {} over labeled QQP (target dataset)".format(model_name))

# Convert the dataset to a DataLoader ready for training
logging.info("Loading BERT labeled QQP dataset")
qqp_train_data = list(InputExample(texts=[data[0], data[1]], label=score) for (data, score) in zip(silver_data, silver_scores))


train_dataloader = DataLoader(qqp_train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.MultipleNegativesRankingLoss(bi_encoder)

###### Classification ######
# Given (quesiton1, question2), is this a duplicate or not?
# The evaluator will compute the embeddings for both questions and then compute
# a cosine similarity. If the similarity is above a threshold, we have a duplicate.
logging.info("Read QQP dev dataset")

dev_sentences1 = []
dev_sentences2 = []
dev_labels = []

with open(os.path.join(qqp_dataset_path, "classification/dev_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        dev_sentences1.append(row['question1'])
        dev_sentences2.append(row['question2'])
        dev_labels.append(int(row['is_duplicate']))

evaluator = BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels,name='dev')

# Configure the training.
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the bi-encoder model
if os.path.exists(bi_encoder_path):
    bi_encoder=SentenceTransformer(bi_encoder_path)
else:
    bi_encoder.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=bi_encoder_path
            )
    bi_encoder = SentenceTransformer(bi_encoder_path)

###############################################################
#
# Evaluate Augmented SBERT performance on QQP benchmark dataset
#
###############################################################


logging.info("Read QQP test dataset")
test_sentences1 = []
test_sentences2 = []
test_labels = []

with open(os.path.join(qqp_dataset_path, "classification/test_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
            test_sentences1.append(row['question1'])
            test_sentences2.append(row['question2'])
            test_labels.append(int(row['is_duplicate']))

evaluator = BinaryClassificationEvaluator(test_sentences1, test_sentences2, test_labels,name='aug_test')
evaluator(bi_encoder, output_path=bi_encoder_path,epoch=0)


'''
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

sbert = SentenceTransformer(modules=[word_embedding_model, pooling_model])
evaluator = BinaryClassificationEvaluator(test_sentences1, test_sentences2, test_labels,name='sbert_test')
evaluator(sbert, output_path=bi_encoder_path,epoch=0)
'''