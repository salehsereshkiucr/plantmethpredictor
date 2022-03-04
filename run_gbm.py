import configs as configs
import data_reader as IP
import gene_body_meth as gbm
import numpy as np
import compatibility as compatibility

cnfig_list = [configs.Cowpea_config, configs.Rice_config, configs.Cucumber_config, configs.Tomato_config]
context_list = ['CG', 'CHG', 'CHH']

for config in cnfig_list:
    methylation_address = config['methylation_address']
    seq_address = config['seq_address']
    annot_address = config['annot_address']
    organism_name = config['organism_name']
    methylations = IP.read_methylations(methylation_address, '')
    annot_df = IP.read_annot(annot_address)
    if organism_name == configs.Cowpea_config['organism_name']:
        methylations = compatibility.cowpea_methylation_compatibility(methylations)
        annot_df = compatibility.cowpea_annotation_compatibility(annot_df)
    genes_df = IP.subset_annot(annot_df, 'gene')
    if organism_name == configs.Rice_config['organism_name']:
        genes_df = IP.subset_annot(annot_df, 'mRNA')
    sequences = IP.readfasta(seq_address)
    if organism_name == configs.Cowpea_config['organism_name']:
        sequences = compatibility.cowpea_sequence_dic_key_compatibility(sequences)
    usls_chrs = list(sequences.keys() - set(genes_df['chr'].unique()))
    if len(usls_chrs) > 0:
        methylations = methylations[methylations.chr.isin(usls_chrs)==False]
        for val in usls_chrs:
            del sequences[val]
    meth_seq = IP.make_meth_string(organism_name, methylations, sequences, 10, from_file=True)
    cntx_seq = IP.make_context_string(organism_name, methylations, sequences, from_file=True)

    for context in context_list:
        genes_avg_p, genes_avg_n, flac_up_avg_p, flac_up_avg_n, flac_down_avg_p, flac_down_avg_n = gbm.get_gene_meth(meth_seq, genes_df,  5, threshold=0.5, context=context, context_seq=cntx_seq)
        final_p = np.concatenate((flac_down_avg_p, genes_avg_p, flac_up_avg_p))
        final_n = np.concatenate((flac_down_avg_n, genes_avg_n, flac_up_avg_n))
        print(final_p)
        print(final_n)
        np.save('./output/' + organism_name + '_' + context + '_final_p.npy', final_p)
        np.save('./output/' + organism_name + '_' + context + '_final_n.npy', final_n)
