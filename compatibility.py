
def arabidopsis_annotation_compatibility(genes_df, sequences):
    chrs = list(sequences.keys())
    genes_df['chr'] = genes_df['chr'].replace(['Chr1'], chrs[0])
    genes_df['chr'] = genes_df['chr'].replace(['Chr2'], chrs[1])
    genes_df['chr'] = genes_df['chr'].replace(['Chr3'], chrs[2])
    genes_df['chr'] = genes_df['chr'].replace(['Chr4'], chrs[3])
    genes_df['chr'] = genes_df['chr'].replace(['Chr5'], chrs[4])
    genes_df['chr'] = genes_df['chr'].replace(['ChrM'], chrs[5])
    genes_df['chr'] = genes_df['chr'].replace(['ChrC'], chrs[6])
    return genes_df

def cowpea_annotation_compatibility(annot_df):
    annot_df = annot_df[~annot_df['chr'].str.contains('contig')]
    return annot_df
def cowpea_methylation_compatibility(methylations):
    methylations = methylations[~methylations['chr'].str.contains('contig')]
    return methylations

def cowpea_chr_fix(chr):
    return chr[:chr.index('(')]

def cowpea_sequence_dic_key_compatibility(seq):
    fixed_seq = {}
    for chr in [y for y in seq.keys() if 'contig' not in y]:
        fixed_seq[cowpea_chr_fix(chr)] = seq[chr]
    return fixed_seq

def rice_annotation_compatibility(annot_df):
    chromosomes = ['NC_008394.4',
                   'NC_008395.2',
                   'NC_008396.2',
                   'NC_008397.2',
                   'NC_008398.2',
                   'NC_008399.2',
                   'NC_008400.2',
                   'NC_008401.2',
                   'NC_008402.2',
                   'NC_008403.2',
                   'NC_008404.2',
                   'NC_008405.2'
                   ]

    annot_df['chr'] = annot_df['chr'].replace([1], chromosomes[0])
    annot_df['chr'] = annot_df['chr'].replace([2], chromosomes[1])
    annot_df['chr'] = annot_df['chr'].replace([3], chromosomes[2])
    annot_df['chr'] = annot_df['chr'].replace([4], chromosomes[3])
    annot_df['chr'] = annot_df['chr'].replace([5], chromosomes[4])
    annot_df['chr'] = annot_df['chr'].replace([6], chromosomes[5])
    annot_df['chr'] = annot_df['chr'].replace([7], chromosomes[6])
    annot_df['chr'] = annot_df['chr'].replace([8], chromosomes[7])
    annot_df['chr'] = annot_df['chr'].replace([9], chromosomes[8])
    annot_df['chr'] = annot_df['chr'].replace([10], chromosomes[9])
    annot_df['chr'] = annot_df['chr'].replace([11], chromosomes[10])
    annot_df['chr'] = annot_df['chr'].replace([12], chromosomes[11])

    return annot_df


chromosomes = ['NC_008394.4',
                   'NC_008395.2',
                   'NC_008396.2',
                   'NC_008397.2',
                   'NC_008398.2',
                   'NC_008399.2',
                   'NC_008400.2',
                   'NC_008401.2',
                   'NC_008402.2',
                   'NC_008403.2',
                   'NC_008404.2',
                   'NC_008405.2'
                   ]

# annot_df['chr'] = annot_df['chr'].replace([1], chromosomes[0])
# annot_df['chr'] = annot_df['chr'].replace([2], chromosomes[1])
# annot_df['chr'] = annot_df['chr'].replace([3], chromosomes[2])
# annot_df['chr'] = annot_df['chr'].replace([4], chromosomes[3])
# annot_df['chr'] = annot_df['chr'].replace([5], chromosomes[4])
# annot_df['chr'] = annot_df['chr'].replace([6], chromosomes[5])
# annot_df['chr'] = annot_df['chr'].replace([7], chromosomes[6])
# annot_df['chr'] = annot_df['chr'].replace([8], chromosomes[7])
# annot_df['chr'] = annot_df['chr'].replace([9], chromosomes[8])
# annot_df['chr'] = annot_df['chr'].replace([10], chromosomes[9])
# annot_df['chr'] = annot_df['chr'].replace([11], chromosomes[10])
# annot_df['chr'] = annot_df['chr'].replace([12], chromosomes[11])
