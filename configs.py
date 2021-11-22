
root = '/home/csgrads/ssere004/Organisms/'
Arabidopsis_config = {
    'methylation_address': '/home/csgrads/ssere004/Organisms/Arabidopsis/SRR3171614_1_trimmed_bismark_bt2.CX_report.txt',
    'seq_address': '/home/csgrads/ssere004/Organisms/Arabidopsis/GCF_000001735.4_TAIR10.1_genomic.fa',
    'annot_address': '/home/csgrads/ssere004/Organisms/Arabidopsis/GCF_000001735.4_TAIR10.1_genomic.gtf',
    'organism_name': 'Arabidopsis',
    'annot_types': ['gene', 'exon', 'CDS']
}

Cowpea_config = {
    'methylation_address': root + 'Cowpea/2010_1_bismark_bt2_pe.CX_report.txt',
    'seq_address': root + 'Cowpea/Cowpea_Genome_1.0.fasta',
    'annot_address': root + 'Cowpea/Vunguiculata_540_v1.2.gene.gff3',
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
    'methylation_address': root + 'Tomato/SRR503393_output_forward_paired_bismark_bt2_pe.CX_report.txt',
    'seq_address': root + 'Tomato/GCF_000188115.4_SL3.0_genomic.fa',
    'annot_address': root + 'Tomato/GCF_000188115.4_SL3.0_genomic.gff',
    'organism_name': 'Tomato',
    'annot_types': ['gene', 'exon', 'CDS']
}

Cucumber_config = {
    'methylation_address': root + 'Cucumber/SRR5430777_1_bismark_bt2_pe.CX_report.txt',
    'seq_address': root + 'Cucumber/GCF_000004075.3_Cucumber_9930_V3_genomic.fa',
    'annot_address': root + 'Cucumber/GCF_000004075.3_Cucumber_9930_V3_genomic.gff',
    'organism_name': 'Cucumber',
    'annot_types': ['gene', 'exon', 'CDS']
}
