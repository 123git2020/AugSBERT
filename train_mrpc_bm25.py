"""
The script shows how to train Augmented SBERT (In-Domain) strategy for STSb dataset with BM25 sampling.
We utlise easy and practical elasticsearch (https://www.elastic.co/) for BM25 sampling.

Installations:
For this example, elasticsearch to be installed (pip install elasticsearch)
[NOTE] You need to also install ElasticSearch locally on your PC or desktop.
link for download - https://www.elastic.co/downloads/elasticsearch
Or to run it with Docker: https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html

Methodology:
Three steps are followed for AugSBERT data-augmentation with BM25 Sampling - 
    1. Fine-tune cross-encoder (BERT) on gold  dataset
    2. Fine-tuned Cross-encoder is used to label on BM25 sampled unlabeled pairs (silver STSb dataset) 
    3. Bi-encoder (SBERT) is finally fine-tuned on both gold + silver  dataset

Citation: https://arxiv.org/abs/2010.08240

Usage:
python train_sts_indomain_bm25.py

OR
python train_sts_indomain_bm25.py pretrained_transformer_model_name top_k

python train_sts_indomain_bm25.py bert-base-uncased 3

"""
from fastbm25 import fastbm25
from torch.utils.data import DataLoader
from sentence_transformers import models, losses, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
import logging
import csv
import numpy as np
import tqdm
import math
import os
import torch
import random

random.seed(1023)
torch.manual_seed(86)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
task = 'mrpc'
model_name = 'bert-base-uncased'
top_k =  2
batch_size = 16
num_epochs = 1
max_seq_length = 128


cross_encoder_path = task+'/_indomain_cross_'+model_name.replace("/", "-")
bi_encoder_path = task+'/_augsbert_BM25_'+model_name.replace("/", "-")
sbert_path = task+'/_sbert_'+model_name.replace("/", "-")


###### Cross-encoder (simpletransformers) ######
logging.info("Loading sentence-transformers model: {}".format(model_name))
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for cross-encoder model
cross_encoder = CrossEncoder(model_name, num_labels=1)


###### Bi-encoder (sentence-transformers) ######
logging.info("Loading bi-encoder model: {}".format(model_name))
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])


#####################################################################
#
# Step 1: Train cross-encoder model with gold dataset
#
#####################################################################

logging.info("Step 1: Train cross-encoder: ({})".format(model_name))

gold_samples = []
dev_samples = []
test_samples = []

with open(task+'/data.tsv', 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = int(row['label'])  # Classification labels are integers

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
evaluator = CEBinaryAccuracyEvaluator.from_input_examples(dev_samples, name='dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the cross-encoder model or load an existing model
if os.path.exists(cross_encoder_path):
    cross_encoder=CrossEncoder(cross_encoder_path)
else:
    cross_encoder.fit(train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            output_path=cross_encoder_path)

############################################################################
#
# Step 2: Label BM25 sampled silver dataset using cross-encoder model
#
############################################################################

#### Top k similar sentences to be retrieved ####
#### Larger the k, bigger the silver dataset ####

logging.info("Step 2.1: Generate silver dataset using top-{} bm25 combinations".format(top_k))

unique_sentences = set()

for sample in gold_samples:
    unique_sentences.update(sample.texts)

unique_sentences = list(unique_sentences) # unique sentences
token2sent = {sentence.lower():sentence for idx,sentence in enumerate(unique_sentences)} #全小写句子到原句的mapping

tokens= [token.split(" ") for token in token2sent.keys()]  #小写句子按空格拆分成token
bm25=fastbm25(tokens)
silver_data = [] 

for t in tokens:
    k_sent=bm25.top_k_sentence(t,k=top_k+1)       #第一个句子就是原句，所以要top(k+1)才能取到k个不同的句子
    if(len(k_sent)==top_k+1):
        silver_data.append((token2sent[" ".join(t)], token2sent[" ".join(k_sent[1][0])]))
        silver_data.append((token2sent[" ".join(t)], token2sent[" ".join(k_sent[2][0])]))



logging.info("Number of silver pairs generated for "+task+": {}".format(len(silver_data)))
logging.info("Step 2.2: Label " + task + "(silver dataset) with cross-encoder: {}".format(model_name))

#cross_encoder = CrossEncoder(cross_encoder_path)
silver_scores = cross_encoder.predict(silver_data)
silver_labels = silver_scores>0.6

#################################################################################################
#
# Step 3: Train bi-encoder model with both (gold + silver) dataset - Augmented SBERT
#
#################################################################################################

logging.info("Step 3: Train bi-encoder: {} with (gold + silver dataset)".format(model_name))

# Convert the dataset to a DataLoader ready for training
logging.info("Read " + task +" gold and silver train dataset")
silver_samples = list(InputExample(texts=[data[0], data[1]], label=score) for \
    data, score in zip(silver_data, silver_labels))


train_dataloader = DataLoader(gold_samples + silver_samples, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model=bi_encoder)     #cosine损失 

logging.info("Read "+task+ " dev dataset")
evaluator = BinaryClassificationEvaluator.from_input_examples(dev_samples, name='dev')

# Configure the training.
warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
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

######################################################################
#
# Evaluate Augmented SBERT performance on STS benchmark (test) dataset
#
######################################################################

# test on the augmented-sbert model or load a existing model
logging.info("Read"+ task +"test dataset")
test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_samples, name='test')
test_evaluator(bi_encoder, output_path=bi_encoder_path,epoch=0)


######################################################################
#
# Evaluate SBERT performance on STS benchmark (test) dataset
#
######################################################################

word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

sbert=SentenceTransformer(modules=[word_embedding_model, pooling_model])
loss = losses.CosineSimilarityLoss(model=sbert)
evaluator = BinaryClassificationEvaluator.from_input_examples(dev_samples, name='dev')
train_dataloader = DataLoader(gold_samples, shuffle=True, batch_size=batch_size)

if os.path.exists(sbert_path):
    sbert=SentenceTransformer(sbert_path)
else:
    sbert.fit(train_objectives=[(train_dataloader, loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=sbert_path
    )

test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_samples, name='test')
test_evaluator(sbert, output_path=sbert_path,epoch=0)
