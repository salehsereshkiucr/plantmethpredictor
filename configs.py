
root = '/home/csgrads/ssere004/Organisms/'
Arabidopsis_config = {
    'methylation_address': '/home/csgrads/ssere004/Organisms/Arabidopsis/SRR3171614_1_trimmed_bismark_bt2.CX_report.txt',
    'seq_address': '/home/csgrads/ssere004/Organisms/Arabidopsis/GCF_000001735.4_TAIR10.1_genomic.fa',
    'annot_address': '/home/csgrads/ssere004/Organisms/Arabidopsis/GCF_000001735.4_TAIR10.1_genomic.gtf',
    'organism_name': 'Arabidopsis',
    'annot_types': ['gene', 'exon', 'CDS']
}

Cowpea_config = {
    'organism_name': 'Cowpea',
    'annot_types': ['gene', 'CDS']

}

Rice_config = {
    'methylation_address': root + 'Rice/SRR618545_final_bismark_bt2.CX_report.txt',
    'seq_address': root + 'Rice/IRGSP-1.0_genome.fasta',
    'annot_address': root + 'Rice/transcripts_exon.gff',
    'organism_name': 'Rice',
    'annot_types': ['mRNA', 'exon']
}

Tomato_config = {
    'organism_name': 'Tomato',
    'annot_types': ['gene', 'exon', 'CDS']
}

Cucumber_config = {
    'organism_name': 'Cucumber',
    'annot_types': ['gene', 'exon', 'CDS']
}
