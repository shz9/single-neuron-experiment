Testing the learning capacity of a single neuron
===

![Alt text](/_images/single_neuron_learning.gif?raw=true "AA Probabilities")

![Alt text](/_images/weight_matrix_adj.gif?raw=true "Weight Matrix")

This repo contains an implementation of a single artificial neuron as well as a
module that's used to test its capacity to learn the genetic code. I was inspired 
to run these experiments by David MacKay's [discussion](https://youtu.be/Z1pcTxvCOgw) 
of the learning capacity of neurons from an information-theoretic perspective.

The genetic code is a simple discrete mapping (i.e. function) between codons and 
amino acids. We can design a neuron to learn this function in any number of ways. 
The way I ended up going about it is I created a neuron with 6 binary inputs and 
1 multiclass output.

**Why a neuron with 6 inputs?**

The reason I chose to design a neuron with 6 inputs is that I wanted the inputs
to be binary (i.e. either 0 or 1). Since there are 4 possible nucleotides in DNA, 
I represented each nucleotide with 2 bits, as follows:

```
codon_to_bits = {
    'T': (0, 0),
    'C': (0, 1),
    'A': (1, 0),
    'G': (1, 1)
}
```

Each codon is made up of 3 nucleotides. Doing the math, we end up with 3 x 2 = 6
binary inputs for the neuron. In this setting, for example, the input 
`(0, 0, 0, 0, 0, 0)` corresponds to the TTT codon, whereas the `(0, 0, 1, 0, 1, 1)`
codon corresponds to the TAG codon.

**The output of the neuron**

As mentioned above, the neuron performs multiclass classification, where each
codon is mapped to one of the 21 classes. The 21 classes correspond to the
20 amino acids, in addition to the stop codon (here represented as `_`): 

```
codon_classes = ['A', 'C', 'E', 'D', 'G', 'F', 'I',
                 'H', 'K', 'M', 'L', 'N', 'Q', 'P',
                 'S', 'R', 'T', 'W', 'V', 'Y', '_']
```

As you may know, the genetic code is "degenerate", meaning multiple codons 
could map to the same class. The mapping is defined in the `codon_table`
dictionary:

```
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
```

**Experiment is still in progress...**