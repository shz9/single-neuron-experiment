"""
Author: Shadi Zabad
Date: October 2017

Experiment is still in progress...
"""

from matplotlib import pylab as plt
from neuron import Neuron
import numpy as np


codon_table = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
    'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
}

codon_classes = ['A', 'C', 'E', 'D', 'G', 'F', 'I',
                'H', 'K', 'M', 'L', 'N', 'Q', 'P',
                'S', 'R', 'T', 'W', 'V', 'Y', '_']


def convert_codon_to_bits(codon_str, codon_to_bits=None):
    """
    DNA is made up of 4 nucleotides:    A, T, C, G.
    These can be represented by 2-bits: 00, 01, 10, 11
    This means that we can represent DNA triplets
    (i.e. codons) with 6 bits, 2 for each position.

    :param codon_str:
    :param codon_to_bits:
    :return:
    """

    if codon_to_bits is None:
        codon_to_bits = {
            'A': (0, 0),
            'T': (0, 1),
            'G': (1, 0),
            'C': (1, 1)
        }

    return (1, ) + codon_to_bits[codon_str[0]] + \
           codon_to_bits[codon_str[1]] + \
           codon_to_bits[codon_str[2]]


def objective_function(neuron, x_input, y_output):
    m = x_input.shape[0]

    curr_weights = neuron.get_weights()

    y_mat = np.zeros(curr_weights.shape[1])
    y_mat[y_output] = 1.0
    class_prob = neuron.activation_function(x_input)

    grad = (-1 / m) * np.dot(x_input.T, (y_mat - class_prob))
    loss = (-1 / m) * np.sum(y_mat * np.log(class_prob).T)

    neuron.set_weights(curr_weights - grad)

    return loss


def test_neuron_accuracy(neuron):

    correct_count = 0
    total = 0

    for cd in codon_table.keys():
        n_out = neuron.get_output(np.matrix(convert_codon_to_bits(cd)))
        if n_out == codon_classes.index(codon_table[cd]):
            correct_count += 1
        total += 1

    return float(correct_count) / total


def plot_codon_prob(ax, codon, codon_probs, true_aa):
    """

    :param codon_probs:
    :param true_aa:
    :return:
    """

    idx, width = np.arange(len(codon_probs)), 0.5

    ax.plot([true_aa, true_aa], [0.0, 1.0], "k--")
    ax.bar(idx, codon_probs, width, color='#40E0D0', align='center')

    ax.set_ylabel('AA Probability')
    ax.set_title(codon)
    ax.set_xticks(idx)
    ax.set_xticklabels(codon_classes)

    ax.set_xlim([min(idx) - 1, max(idx) + 1])


def plot_probs(iter_num):

    fig, ax = plt.subplots(figsize=(30, 18), nrows=8, ncols=8, sharex=True, sharey=True)
    codons = sorted(codon_table.keys())

    for i in range(8):
        for j in range(8):
            cod_prob = single_neuron.activation_function(np.matrix(convert_codon_to_bits(codons[i + j])))
            plot_codon_prob(ax[i, j],
                            codons[i + j],
                            np.squeeze(np.asarray(cod_prob[0])),
                            codon_classes.index(codon_table[codons[i + j]]))

    fig.suptitle("Amino Acid Probabilities - Iteration " + str(iter_num), fontsize=24)
    fig.savefig("./_images/aa_prob_" + str(iter_num) + ".png")

    plt.close()


def plot_weights(iter_num):

    idx = np.arange(len(codon_classes))

    plt.imshow(single_neuron.get_weights(), cmap='GnBu', interpolation='nearest')

    plt.xticks(idx, codon_classes)
    plt.title("Weight Matrix - Iteration " + str(iter_num))

    plt.savefig("./_images/weights_" + str(iter_num) + ".png")

    plt.close()


# A 7 x 21 matrix representing the initial weights
# of the neuron (initially, they're all set to zero).
# NOTE: The first row is the bias.

init_weights = np.matrix([
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
     0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
     0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
     0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
     0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
     0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
     0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
     0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
])

single_neuron = Neuron(init_weights)

print test_neuron_accuracy(single_neuron)

losses = []

for i in range(30):
    plot_probs(i + 1)
    plot_weights(i + 1)
    for cod in codon_table.keys():
        loss = objective_function(single_neuron,
                                  np.matrix(convert_codon_to_bits(cod)),
                                  codon_classes.index(codon_table[cod]))
        losses.append(loss)

print "after:"
print test_neuron_accuracy(single_neuron)
