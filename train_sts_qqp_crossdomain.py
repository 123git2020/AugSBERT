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
import logging
import csv
import torch
import math
import os
import random

random.seed(2)
torch.manual_seed(9)

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
frac=0.2                        #  有多大比例的qqp用作silver label


cross_encoder_path = 'qqp/stsb_indomain_'+model_name.replace("/", "-")
bi_encoder_path = 'qqp/bi-encoder_cross_domain_'+model_name.replace("/", "-")

###### Cross-encoder (simpletransformers) ######

logging.info("Loading cross-encoder model: {}".format(model_name))
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for cross-encoder model
cross_encoder = CrossEncoder(model_name, num_labels=1)

###### Bi-encoder (sentence-transformers) ######

logging.info("Loading bi-encoder model: {}".format(model_name))

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])


#####################################################
#
# Step 1: Train cross-encoder model with STSbenchmark
#
#####################################################

logging.info("Step 1: Train cross-encoder: {} with STSbenchmark (source dataset)".format(model_name))

gold_samples = []
dev_samples = []
test_samples = []

with open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1

        if row['split'] == 'dev':
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        elif row['split'] == 'test':
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        else:
            #As we want to get symmetric scores, i.e. CrossEncoder(A,B) = CrossEncoder(B,A), we pass both combinations to the train set
            gold_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
            gold_samples.append(InputExample(texts=[row['sentence2'], row['sentence1']], label=score))
            

# We wrap gold_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(gold_samples, shuffle=True, batch_size=batch_size)


# We add an evaluator, which evaluates the performance during training
evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='sts-dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the cross-encoder model
if os.path.exists(cross_encoder_path):
    cross_encoder=CrossEncoder(cross_encoder_path)
else:
    cross_encoder.fit(train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=cross_encoder_path)
    cross_encoder = CrossEncoder(cross_encoder_path)

##################################################################
#
# Step 2: Label QQP train dataset using cross-encoder (BERT) model
#
##################################################################

logging.info("Step 2: Label QQP (target dataset) with cross-encoder: {}".format(model_name))

silver_data = []

with open(os.path.join(qqp_dataset_path, "classification/train_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
            if random.random()<frac:
                    silver_data.append([row['question1'], row['question2']])

silver_scores = cross_encoder.predict(silver_data)

# All model predictions should be between [0,1]
assert all(0.0 <= score <= 1.0 for score in silver_scores)

binary_silver_scores = [1 if score >= 0.5 else 0 for score in silver_scores]

###########################################################################
#
# Step 3: Train bi-encoder (SBERT) model with QQP dataset - Augmented SBERT
#
###########################################################################

logging.info("Step 3: Train bi-encoder: {} over labeled QQP (target dataset)".format(model_name))

# Convert the dataset to a DataLoader ready for training
logging.info("Loading BERT labeled QQP dataset")
qqp_train_data = list(InputExample(texts=[data[0], data[1]], label=score) for (data, score) in zip(silver_data, binary_silver_scores))


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