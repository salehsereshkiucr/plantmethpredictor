from Bio import SeqIO
import pandas as pd

def readfasta(address):
    recs = SeqIO.parse(address, "fasta")
    sequences = {}
    for chro in recs:
        sequences[chro.id] = chro.seq
    for i in sequences.keys():
        sequences[i] = sequences[i].upper()
    return sequences

def read_annot(address, chromosomes = None):
    annot_df = pd.read_table(address, sep='\t', comment='#')
    annot_df.columns = ['chr', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
    if chromosomes != None:
        annot_chrs = annot_df.chr.unique()
        for chr in annot_chrs:
            if chr not in chromosomes:
                annot_df = annot_df[annot_df['chr'] != chr]
    return annot_df

def read_methylations(address):
    methylations = pd.read_table(address)
    methylations.columns = ['chr', 'position', 'strand', 'meth', 'unmeth', 'context', 'three']
    return methylations

