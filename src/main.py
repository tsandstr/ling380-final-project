import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

import data
import model

parser = argparse.ArgumentParser(description='Evaluate language model on examples')
parser.add_argument('--model', type=str, default='lstm.pt',
                    help="location of model to evaluate")
parser.add_argument('--evaluate', type=str, default='../sentences.txt',
                    help='location of example sentences to evaluate')
parser.add_argument('--dictionary', type=str, default='dictionary.p',
                    help='location of the dictionary to use')
parser.add_argument('--embedding_size', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--hidden_size', type=int, default=650,
                    help='number of hidden units per layer')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')

args = parser.parse_args()

with open(args.dictionary, 'rb') as f:
    dictionary = pickle.load(f)

model = model.RNNModel(len(dictionary), args.embedding_size, args.hidden_size,
                       args.num_layers, args.dropout, args.tied)

with open(args.model, 'rb') as f:
    model = torch.load(f).to(torch.device("cpu"))
    model.rnn.flatten_parameters()
    model.eval()

def word_to_id(dictionary, word):
    unk_id = dictionary.word2idx['<unk>']
    return dictionary.word2idx.get(word, unk_id)
    
def read_examples_file_as_tensor(examples, dictionary):
    """Reads a text file with one example sentence on each line, and returns a
    tensor of shape (max_len + 1, num_sentences), where max_len is the length of
    the longest sentence (not including an <eos> token) and num_sentences is the
    number of sentences."""
    with open(examples, 'r', encoding="utf8") as f:
        sequences = []
        npi_marker_indexes = []
        seq_len = 0
        for line in f:
            if not line:
                continue

            words = ['<bos>'] + line.split() + ['<eos>']

            if words[1] == '#':
                continue

            if '*' not in words:
                print("Skipping sentence with no NPI marker")
                continue
            
            npi_marker_indexes.append(words.index('*'))
            
            ids = [word_to_id(dictionary, w) for w in words if w != '*']
            sequences.append(ids)
            if len(words) > seq_len:
                seq_len = len(words)

    eos_id = dictionary.word2idx['<eos>']
    sequences = [s + [eos_id] * (seq_len - len(s)) for s in sequences]
    sequences = [torch.tensor(seq).type(torch.int64) for seq in sequences]
    return torch.stack(sequences), npi_marker_indexes
    
def surprisal(model, batch):
    """Calculates the surprisal for each word (with the first word having
    surprisal zero) in a batch of sequences. If batch has shape (num_sequences,
    seq_len), returns a tensor of surprisal values of the same shape.

    """
    initial_hidden = torch.zeros(args.num_layers, batch.size(0),
                                 args.hidden_size)
    with torch.no_grad():
        model_output, _ = model(batch.transpose(0, 1),
                                (initial_hidden,
                                 initial_hidden))
        model_output = model_output.transpose(0, 1)

        # Throw away the first column, because it has no preceding context.
        next_word = torch.narrow(batch, 1, 1, batch.size(1) - 1)
        # Throw away the last column because we don't know what should come next.
        model_output = torch.narrow(model_output, 1, 0, batch.size(1) - 1)
        
        # Squeeze and unsqueeze to make dimensions line up
        next_word = next_word.unsqueeze(-1)
        next_word_log_prob = torch.gather(model_output, -1, next_word)
        next_word_log_prob = next_word_log_prob.squeeze()
        
        # The model returns the natural log of the probability, so we divide by
        # log(2) to get the log base 2.
        surprisal = -next_word_log_prob / torch.log(torch.tensor(2))
        # Set the surprisal of the initial word to be zero
        surprisal = torch.cat([torch.zeros(surprisal.size(0), 1), surprisal], 1)

    return surprisal

def plot_surprisal(vocab, model, batch):
    surp = surprisal(model, batch)

    for i, seq in enumerate(batch):
        seq_surprisal = surp[i]
        seq_surprisal = seq_surprisal.detach().numpy()

        seq_end = seq.tolist().index(vocab.word2idx['<eos>']) + 1
        seq = seq[0:seq_end]
        seq_surprisal = seq_surprisal[0:seq_end]

        seq_words = [vocab.idx2word[word_id] for word_id in seq.tolist()]
        
        series = pd.Series(seq_surprisal, seq_words)
        series.plot()
        plt.xticks(range(len(seq_words)), labels=seq_words)

        plt.show()


def compute_surprisal_on_batch(model, dictionary, batch, markers):
    surp = surprisal(model, batch)
    results = []
    for i, seq in enumerate(batch):
        seq_surprisal = surp[i]
        marker = markers[i]

        seq_end = seq.tolist().index(dictionary.word2idx['<eos>']) + 1
        seq = seq[0:seq_end]
        seq_surprisal = seq_surprisal[0:seq_end]
        
        
        point_surprisal = surp[i, marker].item()

        cumulative_surprisal = 0
        for j in range(marker, seq.size(0)):
            cumulative_surprisal += surp[i, j].item()

        results.append((point_surprisal, cumulative_surprisal))
        
    return results

def compute_licensing_interaction_paired(surprisal_list):
    good = surprisal_list[::2]
    bad = surprisal_list[1::2]
    
    results = []
    for i in range(len(good)):
        good_pnt, good_cum = good[i]
        bad_pnt, bad_cum = bad[i]
        
        results.append(((bad_pnt - good_pnt), (bad_cum - good_cum)))

    return results

def generate_seq(model, dictionary, seed, length):
    seq = seed.split()
    for i in range(length):
        next_word = continue_seq(model, dictionary, seq, 2)[0]
        seq.append(next_word)
        
    return seq

def continue_seq(model, dictionary, seq, choices):
    seq = [dictionary.word2idx[w] for w in seq]
    seq = torch.tensor([seq])

    seq_len = seq.size(-1)
    hidden = model.init_hidden(seq_len)

    logits = model(seq, hidden)[0]
    idx = torch.argsort(logits, descending=True)
    out = idx[:, seq_len - 1, :choices]
    out = out.squeeze().tolist()
    out = [dictionary.idx2word[i] for i in out]

    return out

def foo():
    batch, npi_markers = read_examples_file_as_tensor(args.evaluate, dictionary)
    thing = compute_surprisal_on_batch(model, dictionary, batch, npi_markers)
    thing = compute_licensing_interaction_paired(thing)
    thing = [j for i, j in thing]
    return ttest_1samp(thing, 0, alternative='greater')

def bar():
    batch, npi_markers = read_examples_file_as_tensor(args.evaluate, dictionary)
    thing = compute_surprisal_on_batch(model, dictionary, batch, npi_markers)
    thing = compute_licensing_interaction_paired(thing)
    thing = [j for i, j in thing]
    thing = pd.Series(thing)
    return thing
    
    
batch, npi_markers = read_examples_file_as_tensor(args.evaluate, dictionary)
sns.set()
plot_surprisal(dictionary, model, batch)
