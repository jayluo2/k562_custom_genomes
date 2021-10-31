""" Generate custom diploid reference genome from vcf files. Also contains additional functionality.

vcf2diploid reference:
    Rozowsky J. et al. AlleleSeq: analysis of allele-specific expression and binding in a network framework.
    Mol Syst Biol. 2011.

The author thanks Tao Wang (from the Snyder Lab) for helpful suggestions for code optimization.

TODO: may need to implement hg19-VCF base identity checks (not sure)

"""
import numpy as np
import pandas as pd
import subprocess
import random
import math
import time


def load_vcf(paths, chrom, cols_l):
    """ Obtain VCF files from provided list of paths

    :param paths: [list] paths to vcf files
    :param chrom: [str] chromosome of interest, e.g. 'chr5'
    :return: Dataframe of variants for given chromosome 'chrom' or all chromosomes if chrom == 'all'
    """
    for i in paths:
        if '.vcf' not in i:
            raise FileNotFoundError('Input must be a vcf file!')

    print('Reading and preprocessing vcf files...')

    varied = pd.read_table(paths[0], names=cols_l, low_memory=False, sep='\t')    # various variants
    snps = pd.read_table(paths[1], names=cols_l, low_memory=False, sep='\t')      # mostly SNPs
    dels = pd.read_table(paths[2], names=cols_l, low_memory=False, sep='\t')      # mostly deletions
    bnd = pd.read_table(paths[3], names=cols_l, low_memory=False, sep='\t')        # breakend-represented variants
    varied = process_vcf(varied)
    snps = process_vcf(snps)
    dels = process_vcf(dels)
    bnd = process_vcf(bnd)
    var_all = pd.concat([bnd, varied, dels, snps], axis=0)      # all variants in one dataframe
    var_all['POS'] = var_all['POS'].astype('int64')
    var_all['POS'] = var_all['POS'] - 1                         # use 0-based indexing

    if chrom != 'all':
        var_all = var_all[var_all['#CHROM'] == chrom]     # subset dataframe for given chromosome

    return var_all


def process_vcf(df):
    """ Remove header lines from vcf dataframe

    :param df: Dataframe containing variants, output from load_vcf()
    :return: Dataframe with metadata (header) lines removed
    """
    df = df[~df.iloc[:, 0].str.contains('##')].reset_index(drop=True)
    df.columns = df.iloc[0]
    df = df.drop(0).reset_index(drop=True)

    return df


def load_fasta(fa_dir, chrom):
    """ Load fasta file of reference genome (hg19).

    UCSC chromosome-level fasta files are required (downloaded from https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/)

    :param fa_dir: directory containing all chromosome-level fasta files (i.e. directory containing chr1.fa, chr2.fa, etc.)
    :param chrom: chromosome string (e.g. 'chr1')
    :return: header-excluded reference sequence of the given chromosome
    """
    fa_name = '%s.fa' % chrom
    fa = subprocess.run('ls', stdout=subprocess.PIPE, cwd=fa_dir)     # fasta files list
    fa_l = fa.stdout.decode('utf-8').split('\n')
    if fa_name in fa_l:
        print('fasta file for %s found' % chrom)
        with open(fa_dir + fa_name, 'r') as fa:
            fa_str = fa.read()
    else:
        raise FileNotFoundError('Fasta file for %s cannot be found' % chrom)

    chrom_label = fa_str.split('\n')[0]
    fa_str_final = fa_str.replace(chrom_label, '').replace('\n', '')

    return fa_str_final


def as_dict(seq_str, chr_i):
    """ Returns dictionary that has sequence indices as keys and bases as values.

    :param seq_str: [str] chromosome sequence
    :param chr_i: [str] chromosome name, e.g. 'chr5'
    :return: dictionary mapping base indices to bases (of sequence) for chr_i
    """
    t_1 = time.time()
    seq_dict = {}
    for i in range(len(seq_str)):
        seq_dict[i] = seq_str[i]

    print('Generated sequence dictionary for %s' % chr_i)
    print('as_dict() took %.3f seconds for %s' % (time.time() - t_1, chr_i))

    return seq_dict


def alt_symbols(vcf_df):
    """ Identify entries with symbolic variant representation in the 'ALT' field. Used in vcf_filt().

    Some variants are represented by angled-bracket descriptors like <INS> and <DEL>. These descriptors are
    the ones described in the latest VCF file documentation from https://github.com/samtools/hts-specs

    :param vcf_df: dataframe containing variants
    :return: dataframes containing entries with symbolic variant names
    """
    entry_len = vcf_df['REF'].str.len()
    alt_str = vcf_df['ALT']
    del_labeled = vcf_df[(entry_len == 1) & (alt_str.str.contains('DEL'))]   # symbolic deletion variant
    ins_labeled = vcf_df[(entry_len == 1) & (alt_str.str.contains('INS'))]   # symbolic insertion variant
    inv_labeled = vcf_df[(entry_len == 1) & (alt_str == '<INV>')]   # symbolic inversion variant

    return del_labeled, ins_labeled, inv_labeled


def vcf_filt(vcf_df, vt, chrom):
    """ Filter vcf dataframe based on chromosome and variant type

    Note: tandem duplication appears to be the only type of duplication (relative to reference) for K562

    :param vcf_df: full VCF dataframe (containing data for all chromosomes)
    :param vt: [str] variant type.'SNP' for SNP, 'INS' for insertion, 'DEL' for deletion, 'DUP' for duplication,
    'INV' for inversion, 'BND' for breakend-represented variant
    :param chrom: [str] chromosome name, e.g. 'chr8'
    :return: Dataframe containing variants on chromosome chrom of variant type vt
    """
    filt = 0
    if chrom != 'all':
        vcf_df_filt = vcf_df[vcf_df['#CHROM'] == chrom]
    else:
        vcf_df_filt = vcf_df

    if vt == 'SNP':
        filt = vcf_df_filt[vcf_df_filt['REF'].str.len() == 1]
        filt = filt[filt['ALT'].str.len() == 1]
    elif vt == 'INS':
        filt = vcf_df_filt[vcf_df_filt['REF'].str.len() == 1]
        filt = filt[filt['ALT'].str.len() > 1]                  # not sure if this condition is sufficient
        filt = filt[~filt['ALT'].str.contains('<|>')]
        filt = filt[~filt['ALT'].str.contains('chr')]
        ins_labeled = alt_symbols(vcf_df_filt)[1]
        filt = pd.concat([ins_labeled, filt], axis=0)
    elif vt == 'DEL':
        filt = vcf_df_filt[vcf_df_filt['REF'].str.len() > 1]
        filt = filt[(filt['ALT'].str.len() == 1)]
        filt = filt[~filt['ALT'].str.contains('<|>')]
        filt = filt[~filt['ALT'].str.contains('chr')]
        del_labeled = alt_symbols(vcf_df_filt)[0]
        filt = pd.concat([del_labeled, filt], axis=0)
    elif vt == 'DUP':
        filt = vcf_df_filt[vcf_df_filt['ALT'].str.contains('DUP')]
        filt = filt[~filt['ALT'].str.contains('chr')]
    elif vt == 'INV':
        filt = vcf_df_filt[vcf_df_filt['INFO'].str.contains('SVTYPE=INV')]
        filt = filt[~filt['ALT'].str.contains('<|>')]
        filt = filt[~filt['ALT'].str.contains('chr')]
        inv_labeled = alt_symbols(vcf_df_filt)[2]
        filt = pd.concat([inv_labeled, filt], axis=0)
    elif vt == 'BND':
        filt = vcf_df_filt[vcf_df_filt['INFO'].str.contains('SVTYPE=BND')]

    filt = filt.reset_index(drop=True)      # for row-wise iteration during variant integration

    return filt


def get_mult_alt(vcf_df):
    """ Get dataframe containing entries with multiple ALT variants. Used in mult_alt_df_gt().

    :param vcf_df: full VCF dataframe (containing data for all chromosomes)
    :return: dataframe entries with several (>= 2) ALT variants.
    """
    return vcf_df[vcf_df['ALT'].str.contains(',')].sort_values(by=['POS']).reset_index(drop=True)


def mult_alt_df_gt(vcf_df):
    """ Get maternal and paternal dataframes of variant entries with multiple ALT variants

    :param vcf_df: VCF dataframe (same-chromosome insertions, also contain SNPs that are isolated later)
    :return: insertion dataframes of maternal and paternal variants (ALT variants w/ genotypes assigned)
    """
    mult_alt_df = get_mult_alt(vcf_df)
    m_df, p_df = mult_alt_df.copy(), mult_alt_df.copy()
    n_var = mult_alt_df.shape[0]
    for i in range(n_var):
        gt = mult_alt_df.iloc[i, 9].split(':')[0]
        alt = mult_alt_df.iloc[i, 4].split(',')
        alt1, alt2 = alt[0], alt[1]
        if gt == '1|2':
            m_df.iat[i, 4], p_df.iat[i, 4] = alt2, alt1
        elif gt == '2|1':
            m_df.iat[i, 4], p_df.iat[i, 4] = alt1, alt2
        elif gt == '1/2' or gt == '2/1':
            if rand_ht() == 'm':    # ALT option 1 goes to maternal genome
                m_df.iat[i, 4], p_df.iat[i, 4] = alt1, alt2
            elif rand_ht() == 'p':  # ALT option 1 goes to paternal genome
                m_df.iat[i, 4], p_df.iat[i, 4] = alt2, alt1

    return m_df, p_df


def rand_ht():
    """ Generate random haplotype (maternal or paternal)
    """
    ht = ['m', 'p']      # 'm' for maternal, 'p' for paternal

    return ht[random.randint(0, 1)]


def filt_gt(df):
    """ Filter input dataframe for genotype. Output maternal and paternal dataframes.
    Rows (i.e. entries) of the output dataframes are variants to be applied.
    Random haplotype is given for unphased variants.

    :param df: VCF dataframe to be filtered for genotype.
    :return: filtered maternal and paternal VCF dataframes
    """
    m_l, p_l = [], []       # to store row indices of maternal and paternal entries
    n_var = df.shape[0]     # number of variants described in dataframe df

    for i in range(n_var):
        samp_field = df.iloc[i, 9]
        if ':' in samp_field:    # if sample field has information other than genotype
            gt = samp_field.split(':')[0]
        else:
            gt = samp_field
        if '|' in gt:   # if phased
            if gt == '0|1':
                m_l.append(i)
            elif gt == '1|0':
                p_l.append(i)
            elif gt == '1|1':
                m_l.append(i)
                p_l.append(i)
        elif '/' in gt:     # if unphased
            if rand_ht() == 'm':
                if gt == '0/1':
                    m_l.append(i)
                else:
                    continue
            elif rand_ht() == 'p':
                if gt == '1/0':
                    p_l.append(i)
                else:
                    continue

    # maternal and paternal dataframes
    m_df, p_df = df.iloc[m_l], df.iloc[p_l]

    return m_df.reset_index(drop=True), p_df.reset_index(drop=True)


def merge_df(single_alt_df, mult_alt_df):
    """ Combine dataframes with multiple ALT variants and dataframes with single ALT variants
    e.g. maternal SNPs dataframe (single ALT) + maternal SNPs dataframe (multiple ALT) = merged maternal SNPs dataframe

    :param single_alt_df: single-ALT dataframe
    :param mult_alt_df: multiple-ALT dataframe
    :return: merged dataframe (horizontally concatenated)
    """
    return pd.concat([single_alt_df, mult_alt_df]).sort_values(by=['POS']).reset_index(drop=True)


def filt_bnd(bnd_vcf, chrom):
    """ Separate intrachromosomal from interchromosomal breakend-represented variants

    Single breakend events are treated as deletions (two entries/rows) since we work with one chromosome at a time
    """
    bnd_vcf_chr, bnd_vcf_other = 0, 0
    n_bnd = bnd_vcf.shape[0]    # total number of breakend-represented variants (across all chromosomes)
    if chrom != 'all':
        bnd_vcf_chr = bnd_vcf[bnd_vcf['ALT'].str.contains(chrom + ':')]   # intrachromosomal
        bnd_vcf_other = bnd_vcf[~bnd_vcf['ALT'].str.contains(chrom + ':')]   # interchromosomal
        bnd_vcf_chr = bnd_vcf_chr.reset_index(drop=True)
        bnd_vcf_other = bnd_vcf_other.reset_index(drop=True)

    elif chrom == 'all':
        ind_keep = []
        for i in range(n_bnd):
            chr_i = bnd_vcf.iloc[i, 0]
            alt = bnd_vcf.iloc[i, 4]
            if chr_i + ':' not in alt:
                ind_keep.append(i)
            else:
                continue

        bnd_vcf_chr = bnd_vcf.drop(ind_keep).reset_index(drop=True)
        bnd_vcf_other = bnd_vcf.iloc[ind_keep].reset_index(drop=True)

    return bnd_vcf_chr, bnd_vcf_other


def bnd_subset_gt(bnd_df):
    """ Generate maternal and paternal dataframes for breakend-represented variants.

    :param bnd_df: Dataframe containing breakend-represented variants
    :return: Maternal and paternal dataframes of breakend-represented variants
    """
    n_bnd = bnd_df.shape[0]
    m_df, p_df = pd.DataFrame().copy(), pd.DataFrame().copy()
    for i in range(n_bnd):
        bnd_id = bnd_df.iloc[i, 2]
        if ':' in bnd_id:   # if ID delimiter is ':'
            group_header = bnd_id.split(':')[0]
            filt = bnd_df[bnd_df['ID'].str.contains(group_header + ':')]
            n_var = filt.shape[0]
            if n_var == 2:          # deletion
                filt = filt.sort_values(by=['POS'])
                gt = bnd_gt(filt)
                if gt == 'm':
                    m_df = pd.concat([m_df, filt])
                elif gt == 'p':
                    p_df = pd.concat([p_df, filt])
                elif gt == 'both':
                    m_df = pd.concat([m_df, filt])
                    p_df = pd.concat([p_df, filt])
                else:
                    continue
            elif n_var == 4:          # inversion
                filt = filt.sort_values(by=['POS'])
                gt = bnd_gt(filt)
                if gt == 'm':
                    m_df = pd.concat([m_df, filt])
                elif gt == 'p':
                    p_df = pd.concat([p_df, filt])
                elif gt == 'both':
                    m_df = pd.concat([m_df, filt])
                    p_df = pd.concat([p_df, filt])
                else:
                    continue
        elif '_' in bnd_id and bnd_df.iloc[i, 4] != '<INV>':     # if ID delimiter is '_'
            group_id = bnd_id.split('_')[1]
            filt = bnd_df[bnd_df['ID'].str.contains(group_id + '_')]
            n_var = filt.shape[0]
            if n_var == 2:
                filt = filt.sort_values(by=['POS'])
                gt = bnd_gt(filt)
                if gt == 'm':
                    m_df = pd.concat([m_df, filt])
                elif gt == 'p':
                    p_df = pd.concat([p_df, filt])
                elif gt == 'both':
                    m_df = pd.concat([m_df, filt])
                    p_df = pd.concat([p_df, filt])
                else:
                    continue
            elif n_var == 4:
                filt = filt.sort_values(by=['POS'])
                gt = bnd_gt(filt)
                if gt == 'm':
                    m_df = pd.concat([m_df, filt])
                elif gt == 'p':
                    p_df = pd.concat([p_df, filt])
                elif gt == 'both':
                    m_df = pd.concat([m_df, filt])
                    p_df = pd.concat([p_df, filt])
                else:
                    continue
        elif bnd_df.iloc[i, 4] == '<INV>':
            row = pd.DataFrame(bnd_df.iloc[i, :]).T
            gt = bnd_gt(row)
            if gt == 'm':
                m_df = pd.concat([m_df, row])
            elif gt == 'p':
                p_df = pd.concat([p_df, row])
            elif gt == 'both':
                m_df = pd.concat([m_df, row])
                p_df = pd.concat([p_df, row])
            else:
                continue

    m_df = m_df.drop_duplicates().reset_index(drop=True)
    p_df = p_df.drop_duplicates().reset_index(drop=True)

    return m_df, p_df


def bnd_gt(bnd_df):
    """ Annotate dataframes (with 2 or 4 rows) of breakend-represented variants and return genotype. Used in bnd_subset_gt().

    :param bnd_df: Dataframe of breakend-represented variants.
    :return: Assigned genotype for variant.
    """
    n_bnd = bnd_df.shape[0]
    for i in range(n_bnd):
        samp_field = bnd_df.iloc[i, 9]
        if ':' in samp_field:
            gt = samp_field.split(':')[0]
        else:
            gt = samp_field
        if '|' in gt:
            if gt == '0|1':
                return 'm'
            elif gt == '1|0':
                return 'p'
            elif gt == '1|1':
                return 'both'
        elif '/' in gt:
            if rand_ht() == 'm':
                if gt == '0/1':
                    return 'm'
                else:               # may be implied (not sure)
                    continue
            elif rand_ht() == 'p':
                if gt == '1/0':
                    return 'p'
                else:
                    continue

            # Note that if rand_ht() outputs 'm', if gt is '1/0', no further processing is needed since the allele used in the custom
            # genome will simply be the reference allele. The same logic applies for 'p' with gt of '0/1'.


def ind_bnd_intra(bnd_df):
    """ Record indices of intrachromosomal breakend-represented variants

    There are three types of intrachromosomal breakend-represented variants:
        - 2-entry variants are deletions
        - 4-entry variants are inversions
        - 'Uninterpretable' entries that do not make sense

    0-based coordinates/indices of intrachromosomal variants are stored in a nested list in the form
    [[sta_1, end_1], [sta_2, end_2], ..., [sta_n, end_n]], where sta_n and end_n are the start and end of the nth variant.
    These coordinates can represent the start and end of deletions or inversions.

    *Note: not sure why I didn't do filt.iloc[1, 1] + 1 for end coordinate.

    :param bnd_df: dataframe containing breakend variants
    :return: coordinate list for breakend variants
    """
    bnd_coor = []
    # bnd_df = bnd_df.sort_values(by=['ID'])
    n_bnd = bnd_df.shape[0]
    for i in range(n_bnd - 1):
        bnd_id = bnd_df.iloc[i, 2]
        if ':' in bnd_id:   # if ID delimiter is ':'
            group_header = bnd_id.split(':')[0]
            filt = bnd_df[bnd_df['ID'].str.contains(group_header)].reset_index(drop=True)
            n_var = filt.shape[0]
            if n_var == 2:
                filt = filt.sort_values(by=['POS'])
                coor_sta, coor_end = filt.iloc[0, 1] + 1, filt.iloc[1, 1]   # filt.iloc[0, 1] + 1 because that's where deletion actually starts
                alt_coor = str(filt.iloc[0, 4])
                if (str(coor_end + 1) in alt_coor) and (alt_coor.count(']') == 2):  # 'coor_end + 1' because ALT field coordinate is 1-based
                    bnd_coor.append([coor_sta, coor_end, 'NA'])    # 'NA' for uninterpretable breakend variant
                else:
                    bnd_coor.append([coor_sta, coor_end, 'del'])
            elif n_var == 4:          # inversion
                filt = filt.sort_values(by=['POS'])
                coor_sta, coor_end = filt.iloc[1, 1], filt.iloc[2, 1] + 1
                bnd_coor.append([coor_sta, coor_end, 'inv'])
            else:
                continue
        elif '_' in bnd_id:     # if ID delimiter is '_'
            group_id = bnd_id.split('_')[1]
            filt = bnd_df[bnd_df['ID'].str.contains(group_id)]
            n_var = filt.shape[0]
            if n_var == 2:
                filt = filt.sort_values(by=['POS'])
                coor_sta, coor_end = filt.iloc[0, 1] + 1, filt.iloc[1, 1]   # removed -1 from filt.iloc[1, 1]
                alt_coor = filt.iloc[0, 4]
                if (str(coor_end) in alt_coor) and (alt_coor.count(']') == 2):
                    bnd_coor.append([coor_sta, coor_end, 'NA'])    # 'NA' for uninterpretable breakend variant
                else:
                    bnd_coor.append([coor_sta, coor_end, 'del'])
            elif n_var == 4:
                filt = filt.sort_values(by=['POS'])
                coor_sta, coor_end = filt.iloc[1, 1], filt.iloc[2, 1] + 1
                bnd_coor.append([coor_sta, coor_end, 'inv'])
            else:
                continue
        elif bnd_df.iloc[i, 4] == '<INV>':
            info_end = [i for i in bnd_df.iloc[i, 7].split(';') if 'END' in i and ',' not in i]
            coor_sta, coor_end = bnd_df.iloc[i, 1], int(info_end[0].split('=')[1]) + 1
            bnd_coor.append([coor_sta, coor_end, 'inv'])

    bnd_coor_new = [i for i in bnd_coor if i[2] != 'NA']

    # print('%d unrecognized intrachromosomal variants' % len([i for i in bnd_coor if i[2] == 'NA']))

    return bnd_coor_new


def ind_ins(ins_df):
    """ Record insertion indices (all new bases will be marked as 'ins' in the index list
    to avoid changing original base indices)

    Returns nested list in the same format described in the docstring of ind_bnd_intra()
    """
    ins_coor = []
    n_ins = ins_df.shape[0]     # number of insertions (for chromosome)
    for i in range(n_ins):
        coor = ins_df.iloc[i, 1]
        seq = ins_df.iloc[i, 4]     # incl. base preceding actual insertion
        ins_coor.append([coor, seq])

    return ins_coor


def ind_del(del_df):
    """ Record indices of deleted bases and remove overlapping deletions, as well as deletions overlapping with
     insertions or breakend-represented variants
    """
    del_coor = []
    n_del = del_df.shape[0]
    for i in range(n_del):
        alt = del_df.iloc[i, 4]
        if alt == '<DEL>':
            coor_sta = del_df.iloc[i, 1] + 1
            info_end = [i for i in del_df.iloc[i, 7].split(';') if 'END' in i and ',' not in i]
            coor_end = int(info_end[0].split('=')[1]) + 1   # '+ 1' for ease of integration
            del_coor.append([coor_sta, coor_end])
        else:
            coor_sta = del_df.iloc[i, 1] + 1
            coor_end = coor_sta + (len(del_df.iloc[i, 3])) - 1
            del_coor.append([coor_sta, coor_end])

    return del_coor


def ind_snps(snps_df):
    """ Record indices of SNPs and filter said indices based on overlap with previously integrated variants
    """
    snps_coor = []
    n_snps = snps_df.shape[0]
    for i in range(n_snps):
        coor = snps_df.iloc[i, 1]
        snp = snps_df.iloc[i, 4]
        snps_coor.append([coor, snp])

    return snps_coor


def ind_dups(dups_df):
    """ Record start and end coordinates of duplications (all are tandem in the K562 variant data)
    """
    dup_coor = []
    n_dups = dups_df.shape[0]
    for i in range(n_dups):
        alt = dups_df.iloc[i, 4]
        if alt == '<DUP:TANDEM>':
            coor_sta = dups_df.iloc[i, 1] + 1
            info_end = [i for i in dups_df.iloc[i, 7].split(';') if 'END' in i and ',' not in i]
            coor_end = int(info_end[0].split('=')[1])
            dup_coor.append([coor_sta, coor_end])
        else:
            coor_sta = dups_df.iloc[i, 1] + 1
            coor_end = coor_sta + (len(dups_df.iloc[i, 3]))
            dup_coor.append([coor_sta, coor_end])

    return dup_coor


def rm_ind(coor_l, ind_l):
    """ Remove elements from given list
    """
    return [i for ind, i in enumerate(coor_l) if ind not in ind_l]


def coor_rm_intra(coor_inter, coor_intra):
    """ Identify and remove intrachromosomal variant entries from coordinate list
    Output is used to filter dataframe (for intrachromosomal rearrangements) in process_used_ind()
    """
    intra_reg_l = []
    intra_rm_ind = []
    for ind, reg in enumerate(coor_intra):
        reg_sta, reg_end, bnd_type = reg[0], reg[1], reg[2]
        intra_reg_l.append([reg_sta, reg_end])
        for bnd in coor_inter:
            coord = bnd[1]
            if bnd_type == 'del':
                if reg_sta <= coord <= reg_end:
                    intra_rm_ind.append(ind)

    # overlapping intrachromosomal variants are considered during variant integration

    intra_coor_filt = rm_ind(coor_intra, intra_rm_ind)

    return intra_coor_filt, intra_rm_ind


def coor_rm_ins(intra_coor, ins_coor):
    """ Identify and remove insertion variant coordinates from coordinate list

    Note that insertions with different start coordinates cannot overlap because added bases are novel
    relative to the reference genome. Added bases at different start sites do not have reference genome coordinates.
    If two or more insertions have the same start coordinates, only keep the first occurrence.

    :param intra_coor: filtered nested list of region coordinates for intrachromosomal rearrangements
    :param ins_coor: unfiltered nested list of insertion coordinates
    :return:
    """
    ins_rm_ind = []
    ins_coor_final = []
    for ind, coor in enumerate(ins_coor):
        coor_i = coor[0]
        for reg in intra_coor:
            reg_sta, reg_end, bnd_type = reg[0], reg[1], reg[2]
            if bnd_type == 'del':
                if reg_sta < coor_i < reg_end:
                    ins_rm_ind.append(ind)

    ins_coor_filt = rm_ind(ins_coor, ins_rm_ind)
    ins_coor_filt_df = pd.DataFrame(ins_coor_filt)
    ins_coor_filt_df.columns = ['coor', 'seq']
    ins_coor_filt_df = ins_coor_filt_df.drop_duplicates(subset='coor')

    for i in range(ins_coor_filt_df.shape[0]):
        coor, seq = ins_coor_filt_df.iloc[i, 0], ins_coor_filt_df.iloc[i, 1]
        ins_coor_final.append([coor, seq])

    return ins_coor_final, ins_rm_ind


def coor_rm_del(inter_coor, ins_coor, del_coor):
    """ Identify and remove deletion variant coordinates from coordinate list

    Deletions can overlap with other deletions because deleted bases are
    lost relative to the reference genome, and these deleted bases have
    reference genome coordinates. A deletion cannot occur in an already-deleted region
    (relative to the reference genome).

    :param ins_coor: filtered nested list of insertion coordinates
    :param del_coor: unfiltered nested list of deletion coordinates
    :return:
    """
    del_reg_l = []
    del_rm_ind = []
    for ind, reg_i in enumerate(del_coor):
        reg_i_sta, reg_i_end = reg_i[0], reg_i[1]
        del_reg_l.append([reg_i_sta, reg_i_end])
        for coor in ins_coor:        # check overlap with insertions
            if reg_i_sta <= coor[0] <= reg_i_end:
                del_rm_ind.append(ind)
        for coor_i in inter_coor:
            coord = coor_i[1]
            if reg_i_sta <= coord <= reg_i_end:
                del_rm_ind.append(ind)

    # overlapping deletions are addressed during integration, and so is overlap with intrachromosomal rearrangements

    del_coor_filt = rm_ind(del_coor, del_rm_ind)

    return del_coor_filt, del_rm_ind


def filt_df(df, ind_rm_l):
    """ Filter dataframe
    """
    return df.drop(ind_rm_l)


def process_used_ind(intrachr_df, ins_df, del_df, intrachr_rm, ins_rm, del_rm):
    """ Collate all 'used' indices from ind_bnd_inter(), ind_bnd_intra(), ind_ins(), and ind_del()
    These indices are updated for every chromosome.

    Note that the above functions account for (keep in mind the order of integration):
        - Intrachromosomal breakend-represented variants and deletions overlapping with (single) coordinates of
        interchromosomal rearrangements
        - Insertions overlapping with intrachromosomal breakend-represented variants
        - Deletions overlapping with insertions
    The rest of the overlap possibilites are addressed during variant integration:
        - Overlapping intrachromosomal rearrangements
        - Deletions overlapping with intrachromosomal breakend-represented variants
            - Deleted regions cannot be deleted again
            - Inverted regions' reference coordinates are reversed and will not be recognized
            by deletion reference coordinates. Note that SNPs in inversions are not applied
            since the base identities (relative to reference genome) have changed (to the
            complement bases).
        - Deletions overlapping with another/other deletion(s)
        - SNPs overlapping with deletions

    Filtering is sequential:
    1) Remove any intrachromosomal rearrangmenets overlapping with interchromosomal rearrangement coordinates
    2) Remove any indels overlapping with any rearrangement
    3) Remove any SNPs overlapping with any rearrangement or deletions (not insertions since they add new bases)
    """
    intrachr_df_filt = filt_df(intrachr_df, intrachr_rm)
    ins_df_filt = filt_df(ins_df, ins_rm)
    del_df_filt = filt_df(del_df, del_rm)

    return intrachr_df_filt, ins_df_filt, del_df_filt


def as_np_array(seq_str):
    return np.array(list(seq_str), dtype=object)


def to_str(final_seq):
    """ Output some DNA sequence as a string
    """
    return ''.join(final_seq)


def seq_comp(seq):
    """ Get complement sequence of some sequence
    seq is a list, each element a base
    """
    new_seq = []
    for i in seq:
        if i == 'A':
            new_seq.append('T')
        elif i == 'T':
            new_seq.append('A')
        elif i == 'C':
            new_seq.append('G')
        elif i == 'G':
            new_seq.append('C')
        elif i == 'a':
            new_seq.append('T')
        elif i == 't':
            new_seq.append('A')
        elif i == 'c':
            new_seq.append('G')
        elif i == 'g':
            new_seq.append('C')
        else:
            new_seq.append(i)

    return new_seq


def check_labels(seq_l, sta, end):
    subseq = seq_l[sta:end+1]
    # labels_any = len([i for i in subseq if (type(i) == str and i.split('_')[1]) in ['del', 'inv', 'dup']])
    labels_any = len([i for i in subseq if '_' not in i])
    if labels_any == 0:
        return 'OK'
    else:
        raise ValueError('Error')


def search(arr, x):
    """ Modified binary search.
    Most of the code in this function is from https://www.geeksforgeeks.org/python-program-for-binary-search/
    """
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (high + low) // 2
        if type(arr[mid]) == str:
            num = int(arr[mid].split('_')[0])
            if num <= x:
                low = mid + 1
            elif num > x:
                high = mid - 1
        else:
            if arr[mid] < x:
                low = mid + 1
            elif arr[mid] > x:
                high = mid - 1
            else:
                return mid
    return 'NaN'


def base_to_list_ind(seq_ind_l, ref_sta, ref_end):
    """ Convert base coordinate to list index (position of base coordinate in the coordinate list)
    """
    if ref_sta == ref_end:
        if ref_sta in seq_ind_l:
            # a = time.time()
            ind_ref_coor = search(seq_ind_l, ref_sta)   # find list index of reference coordinate
            # print('search(): %.8f' % (time.time() - a))
            return ind_ref_coor
        else:
            return 'NaN'
    elif ref_sta < ref_end:
        if (ref_sta in seq_ind_l) and (ref_end in seq_ind_l):
            ind_ref_coor_sta, ind_ref_coor_end = search(seq_ind_l, ref_sta), search(seq_ind_l, ref_end)
            if check_labels(seq_ind_l, ind_ref_coor_sta, ind_ref_coor_end) == 'OK':
                return ind_ref_coor_sta, ind_ref_coor_end
        else:
            return 'NaN', 'NaN'
    else:
        raise ValueError('Invalid variant coordinates. Variant start and end coordinates can be the same '
                         '(one coordinate). Otherwise, the end coordinate must be greater than the start coordinate.')


def check_dict_key(dict_seq, ref_sta, ref_end):
    if ref_sta == ref_end:
        info = dict_seq[ref_sta]
        status = info[1]
        if status == 'None':    # if no variant has been applied to base 'ref_sta'
            return 'OK'
        else:
            return 'NaN'
    elif ref_sta < ref_end:
        for i in range(ref_sta, ref_end + 1):
            info_i = dict_seq[i]
            status_i = info_i[1]
            if status_i == 'None':
                return 'OK'
            else:
                return 'NaN'
    else:
        raise ValueError('Invalid variant coordinates. Variant start and end coordinates can be the same '
                         '(one coordinate). Otherwise, the end coordinate must be greater than the start coordinate.')


def process_intrachr_dict(fa_dict, intra_coor_filt):
    """

    :param fa_dict: FASTA sequence dictionary
    :param intra_coor_filt: filtered coordinates for intrachromosomal variants (breakend-represented deletions or inversions)
    :return: Updated FASTA dictionary containing the variants
    """
    t_1 = time.time()
    coor_df = pd.DataFrame(intra_coor_filt).drop_duplicates().reset_index(drop=True)
    n_var = coor_df.shape[0]
    num_processed, num_discarded = 0, 0
    for i in range(n_var):
        row = coor_df.iloc[i]
        sta, end, bnd_type = row[0], row[1], row[2]
        if bnd_type == 'del':
            for j in range(sta, end):   # changed 'end + 1' to 'end'
                try:
                    del fa_dict[j]
                except KeyError:
                    num_discarded += 1
                    break
        elif bnd_type == 'inv':
            try:
                inv_bases = [fa_dict[sta]]
            except KeyError:
                continue
            for j in range(sta+1, end):
                try:
                    inv_bases.append(fa_dict[j])
                    del fa_dict[j]
                except KeyError:
                    num_discarded += 1
                    break
            fa_dict[sta] = ''.join(seq_comp(inv_bases))[::-1] + '*'   # '*' to mark modification

        num_processed += 1

    print('Processed %d BND-represented intrachromosomal variants'
          '(%d discarded due to overlaps)' % (num_processed, num_discarded))

    print('process_intrachr(): %.3f seconds' % (time.time() - t_1))

    return fa_dict


def process_dups_dict(fa_dict, dups_coor):
    """ Integrate duplications

    :param fa_dict: FASTA dictionary (from output of process_intrachr_dict())
    :param dups_coor: cooordinates for duplications
    :return: Updated FASTQ dictionary
    """
    t_1 = time.time()
    num_processed = 0
    for i in dups_coor:
        dup_sta, dup_end = i[0], i[1]
        try:
            dup_bases = [fa_dict[dup_sta]]
        except KeyError:
            continue
        for j in range(dup_sta+1, dup_end):
            try:
                dup_bases.append(fa_dict[j])
                del fa_dict[j]
                num_processed += 1
            except KeyError:
                break
        fa_dict[dup_sta] = ''.join(dup_bases)*2 + '*'       # '*' indicates this base has been changed

    print('Processed %d duplications' % num_processed)
    print('process_dups(): %.3f' % (time.time() - t_1))

    return fa_dict


def process_ins_dict(fa_dict, ins_coor_filt):
    """ Integrate insertions.

    :param fa_l: FASTA dictionary (from output of process_dups_dict())
    :param ins_coor_filt: Filtered insertion coordinates
    :return: Updated FASTA dictionary
    """
    t_1 = time.time()
    num_processed, num_discarded = 0, 0
    for i in ins_coor_filt:
        coor, ins_seq = i[0], i[1]      # 'ins_seq' includes base at 'coor' for ease of integration
        try:
            if '*' not in fa_dict[coor]:        # if base has not already been modified
                fa_dict[coor] = ins_seq
            else:
                continue
        except KeyError:
            num_discarded += 1
            continue
        num_processed += 1

    print('Processed %d insertions (removed %d due to overlaps)' % (num_processed, num_discarded))
    print('process_ins(): %.3f' % (time.time() - t_1))

    return fa_dict


def process_del_dict(fa_dict, del_coor_filt):
    """ Integrate non-breakend deletions.

    :param fa_dict: FASTA dictionary (from output of process_ins_dict())
    :param del_coor_filt: Deletion (not breakend-represented) coordinates
    :return: Updated FASTA dictionary
    """
    t_1 = time.time()
    num_processed, num_discarded = 0, 0
    for i in del_coor_filt:
        del_sta, del_end = i[0], i[1]
        for j in range(del_sta, del_end):   # changed 'del_end + 1' to 'del_end'
            try:
                if '*' not in fa_dict[j]:
                    del fa_dict[j]
                else:
                    break
                num_processed += 1
            except KeyError:
                num_discarded += 1
                num_processed += 1
                break

    print('Processed %d deletions (removed %d due to overlaps)' % (num_processed, num_discarded))
    print('process_del_dict(): %.3f' % (time.time() - t_1))

    return fa_dict


def process_snps_dict(fa_dict, snps_coor_filt):
    """ Integrate SNPs

    :param fa_dict: FASTA dictionary (from output of process_del_dict())
    :param snps_coor_filt: Filtered SNP coordinates
    :return:
    """
    t_1 = time.time()

    num_processed, num_discarded = 0, 0
    for ind, i in enumerate(snps_coor_filt):
        coor, snp = i[0], i[1]
        try:
            if '*' not in fa_dict[coor]:
                fa_dict[coor] = snp
                num_processed += 1
            else:
                num_processed += 1
                continue
        except KeyError:    # if 'coor' no longer key of fa_dict, i.e. if base at 'coor' has already been deleted
            num_discarded += 1
            num_processed += 1
            continue

    print('Integrated %d SNPs (%d discarded)' % (num_processed, num_discarded))
    print('process_snps_dict(): %.3f' % (time.time() - t_1))

    return fa_dict


def dict_bases_to_str(fa_dict):
    """ Convert dictionary sequence to string
    """
    t_1 = time.time()
    final_str = ''.join(list(fa_dict.values())).replace('*', '')
    print('dict_bases_to_str(): %.3f' % (time.time() - t_1))
    return final_str


def test_fasta(length):
    """ Generate artificial fasta files to test code
    """
    seq = []
    i = 0
    bases = ['A', 'T', 'C', 'G']
    # random.seed(42)
    while len(seq) < length:
        ind = bases[random.randint(0, 3)]
        seq.append(ind)      # random base index
        i += 1

    return to_str(seq)


def make_fa(output_dir, filename, num_lines, chr_i):
    """ Generate artificial fasta file, output file written to output_dir
    """
    with open(output_dir + filename, 'w') as vcf_test:
        i = 0
        vcf_test.write('>%s\n' % chr_i)
        while i < num_lines:
            vcf_test.write(test_fasta(50) + '\n')
            i += 1


def fa_chr_dict(fa_str_chr, outdir, ht, chr_i):
    """ Generate chromosomal fasta file from final chromosomal, variant-containing fasta sequence string
    """
    if chr_i == 'chr1':
        with open(outdir + 'k562_custom_%s.fa' % ht, 'w') as custom:
            custom.write('>%s\n' % chr_i)
            for ind, i in enumerate(fa_str_chr):
                custom.write(i)
                if (ind + 1) % 50 == 0:
                    custom.write('\n')
    else:
        with open(outdir + 'k562_custom_%s.fa' % ht, 'a') as custom:
            custom.write('\n>%s\n' % chr_i)
            for ind, i in enumerate(fa_str_chr):
                custom.write(i)
                if (ind + 1) % 50 == 0:
                    custom.write('\n')


def num_var(interchr, intrachr, ins, dels, snps, dups, ht):
    interchr_new = pd.DataFrame(interchr).drop_duplicates().reset_index(drop=True)
    intrachr_new = pd.DataFrame(intrachr).drop_duplicates().reset_index(drop=True)
    ins_new = pd.DataFrame(ins).drop_duplicates().reset_index(drop=True)
    dels_new = pd.DataFrame(dels).drop_duplicates().reset_index(drop=True)
    snps_new = pd.DataFrame(snps).drop_duplicates().reset_index(drop=True)
    dups_new = pd.DataFrame(dups).drop_duplicates().reset_index(drop=True)

    intrachr_del = intrachr_new[intrachr_new.iloc[:, 2] == 'del']
    intrachr_unrecognized = intrachr_new[intrachr_new.iloc[:, 2] == 'NA']

    interchr_num = interchr_new.shape[0]
    ins_num = ins_new.shape[0]
    del_num = dels_new.shape[0] + intrachr_del.shape[0]
    snps_num = snps_new.shape[0]
    dups_num = dups_new.shape[0]
    inv_num = intrachr_new[intrachr_new.iloc[:, 2] == 'inv'].shape[0]
    unrecognized_num = intrachr_unrecognized.shape[0]

    print('%s:' % ht)
    print('# of interchromosomal: %d' % interchr_num)
    print('# of duplications: %d' % dups_num)
    print('# of insertions: %d' % ins_num)
    print('# of deletions: %d' % del_num)
    print('# of inversions: %d' % inv_num)
    print('# of SNPs: %d' % snps_num)
    # print('# of unrecognized intrachromosomal: %d' % unrecognized_num)


def get_stats():
    vcf_dir = '/Users/jayluo/Research_all/Snyder_Lab/vcf_ENCODE/vcf_files/unzipped_zipped/'
    f1, f2, f3, f4 = 'ENCFF574MDJ_varied.vcf', 'ENCFF752OAX_SNP.vcf', 'ENCFF785JVR_DEL.vcf', 'ENCFF863MPP_BND.vcf'
    paths = [vcf_dir + f1, vcf_dir + f2, vcf_dir + f3, vcf_dir + f4]
    cols = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'K562']

    chrom = 'all'

    var_df_all = load_vcf(paths, chrom, cols)

    snps_only = vcf_filt(var_df_all, 'SNP', chrom)
    ins_only = vcf_filt(var_df_all, 'INS', chrom)
    del_only = vcf_filt(var_df_all, 'DEL', chrom)
    dup_only = vcf_filt(var_df_all, 'DUP', chrom)
    inv_only = vcf_filt(var_df_all, 'INV', chrom)       # TODO: check why it's not used (error?) [done]
    bnd_only = vcf_filt(var_df_all, 'BND', chrom)
    bnd_intrachromosomal = filt_bnd(bnd_only, chrom)[0]
    bnd_interchromosomal = filt_bnd(bnd_only, chrom)[1]

    # Add '<INV>' alleles to bnd_intrachromosomal
    bnd_intrachromosomal = pd.concat([bnd_intrachromosomal, inv_only]).reset_index(drop=True)

    # Get maternal and paternal dataframes
    bnd_inter_maternal, bnd_inter_paternal = bnd_subset_gt(bnd_interchromosomal)
    bnd_intra_maternal, bnd_intra_paternal = bnd_subset_gt(bnd_intrachromosomal)

    ins_mult_alt_m, ins_mult_alt_p = mult_alt_df_gt(ins_only)   # still containing some SNPs
    snps_mult_alt_m = vcf_filt(ins_mult_alt_m, 'SNP', chrom)
    snps_mult_alt_p = vcf_filt(ins_mult_alt_p, 'SNP', chrom)

    ins_maternal, ins_paternal = filt_gt(ins_only)
    ins_only_mult_alt_m, ins_only_mult_alt_p = vcf_filt(ins_mult_alt_m, 'INS', chrom), vcf_filt(ins_mult_alt_p, 'INS', chrom)
    ins_merged_m, ins_merged_p = merge_df(ins_maternal, ins_only_mult_alt_m), merge_df(ins_paternal, ins_only_mult_alt_p)

    # Deletions
    del_maternal, del_paternal = filt_gt(del_only)

    # SNPs
    snps_maternal, snps_paternal = filt_gt(snps_only)
    snps_merged_m, snps_merged_p = merge_df(snps_maternal, snps_mult_alt_m), merge_df(snps_paternal, snps_mult_alt_p)

    # Duplications
    dups_maternal, dups_paternal = filt_gt(dup_only)

    # Get coordinates for maternal and paternal dataframes
    coord_inter_m, coord_inter_p = ind_bnd_inter(bnd_inter_maternal), ind_bnd_inter(bnd_inter_paternal)
    coord_intra_m, coord_intra_p = ind_bnd_intra(bnd_intra_maternal), ind_bnd_intra(bnd_intra_paternal)
    coord_ins_m, coord_ins_p = ind_ins(ins_merged_m), ind_ins(ins_merged_p)
    coord_del_m, coord_del_p = ind_del(del_maternal), ind_del(del_paternal)
    coord_snps_m, coord_snps_p = ind_snps(snps_merged_m), ind_snps(snps_merged_p)
    coord_dups_m, coord_dups_p = ind_dups(dups_maternal), ind_dups(dups_paternal)

    # Get filtered coordinates
    coord_intra_filt_m, rm_ind_intra_m = coor_rm_intra(coord_inter_m, coord_intra_m)
    coord_intra_filt_p, rm_ind_intra_p = coor_rm_intra(coord_inter_p, coord_intra_p)

    coord_ins_filt_m, rm_ind_ins_m = coor_rm_ins(coord_intra_filt_m, coord_ins_m)
    coord_ins_filt_p, rm_ind_ins_p = coor_rm_ins(coord_intra_filt_p, coord_ins_p)

    coord_del_filt_m, rm_ind_del_m = coor_rm_del(coord_inter_m, coord_ins_filt_m, coord_del_m)
    coord_del_filt_p, rm_ind_del_p = coor_rm_del(coord_inter_p, coord_ins_filt_p, coord_del_p)

    num_var(coord_inter_m, coord_intra_m,
            coord_ins_m, coord_del_m, coord_snps_m, coord_dups_m, 'maternal')

    num_var(coord_inter_p, coord_intra_p,
            coord_ins_p, coord_del_p, coord_snps_p, coord_dups_p, 'paternal')


def run_functions_dict(paths, path_fa_dir, ht, outdir_fasta):
    # vcf_dir = '/Users/jayluo/Research_all/Snyder_Lab/vcf_ENCODE/vcf_files/unzipped_zipped/'
    # f1, f2, f3, f4 = 'ENCFF574MDJ_varied.vcf', 'ENCFF752OAX_SNP.vcf', 'ENCFF785JVR_DEL.vcf', 'ENCFF863MPP_BND.vcf'
    # paths = [vcf_dir + f1, vcf_dir + f2, vcf_dir + f3, vcf_dir + f4]
    cols = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'K562']
    # outdir_fasta = '/Users/jayluo/Research_all/Snyder_Lab/vcf_ENCODE/custom_program_output/'


    for i in range(22):
        if (i + 1) != 23:
            chrom = 'chr%d' % (i + 1)
        else:
            chrom = 'chrX'

        var_df_all = load_vcf(paths, chrom, cols)
        snps_only = vcf_filt(var_df_all, 'SNP', chrom)
        ins_only = vcf_filt(var_df_all, 'INS', chrom)
        del_only = vcf_filt(var_df_all, 'DEL', chrom)
        dup_only = vcf_filt(var_df_all, 'DUP', chrom)
        inv_only = vcf_filt(var_df_all, 'INV', chrom)       # TODO: check why it's not used (error?) [done]
        bnd_only = vcf_filt(var_df_all, 'BND', chrom)
        bnd_intrachromosomal = filt_bnd(bnd_only, chrom)[0]
        bnd_interchromosomal = filt_bnd(bnd_only, chrom)[1]

        # Add '<INV>' alleles to bnd_intrachromosomal
        bnd_intrachromosomal = pd.concat([bnd_intrachromosomal, inv_only]).reset_index(drop=True)

        # Get maternal and paternal dataframes
        bnd_inter_maternal, bnd_inter_paternal = bnd_subset_gt(bnd_interchromosomal)
        bnd_intra_maternal, bnd_intra_paternal = bnd_subset_gt(bnd_intrachromosomal)

        ins_mult_alt_m, ins_mult_alt_p = mult_alt_df_gt(ins_only)   # still containing some SNPs
        snps_mult_alt_m = vcf_filt(ins_mult_alt_m, 'SNP', chrom)
        snps_mult_alt_p = vcf_filt(ins_mult_alt_p, 'SNP', chrom)

        ins_maternal, ins_paternal = filt_gt(ins_only)
        ins_only_mult_alt_m, ins_only_mult_alt_p = vcf_filt(ins_mult_alt_m, 'INS', chrom), vcf_filt(ins_mult_alt_p, 'INS', chrom)
        ins_merged_m, ins_merged_p = merge_df(ins_maternal, ins_only_mult_alt_m), merge_df(ins_paternal, ins_only_mult_alt_p)

        # Deletions
        del_maternal, del_paternal = filt_gt(del_only)

        # SNPs
        snps_maternal, snps_paternal = filt_gt(snps_only)
        snps_merged_m, snps_merged_p = merge_df(snps_maternal, snps_mult_alt_m), merge_df(snps_paternal, snps_mult_alt_p)

        # Duplications
        dups_maternal, dups_paternal = filt_gt(dup_only)

        # Get coordinates for maternal and paternal dataframes
        coord_inter_m, coord_inter_p = ind_bnd_inter(bnd_inter_maternal), ind_bnd_inter(bnd_inter_paternal)
        coord_intra_m, coord_intra_p = ind_bnd_intra(bnd_intra_maternal), ind_bnd_intra(bnd_intra_paternal)
        coord_ins_m, coord_ins_p = ind_ins(ins_merged_m), ind_ins(ins_merged_p)
        coord_del_m, coord_del_p = ind_del(del_maternal), ind_del(del_paternal)
        coord_snps_m, coord_snps_p = ind_snps(snps_merged_m), ind_snps(snps_merged_p)
        coord_dups_m, coord_dups_p = ind_dups(dups_maternal), ind_dups(dups_paternal)

        # Get filtered coordinates
        coord_intra_filt_m, rm_ind_intra_m = coor_rm_intra(coord_inter_m, coord_intra_m)
        coord_intra_filt_p, rm_ind_intra_p = coor_rm_intra(coord_inter_p, coord_intra_p)

        coord_ins_filt_m, rm_ind_ins_m = coor_rm_ins(coord_intra_filt_m, coord_ins_m)
        coord_ins_filt_p, rm_ind_ins_p = coor_rm_ins(coord_intra_filt_p, coord_ins_p)

        coord_del_filt_m, rm_ind_del_m = coor_rm_del(coord_inter_m, coord_ins_filt_m, coord_del_m)
        coord_del_filt_p, rm_ind_del_p = coor_rm_del(coord_inter_p, coord_ins_filt_p, coord_del_p)

        # Following block necessary?
        # intra_filt_df_m, ins_filt_df_m, del_filt_df_m = process_used_ind(bnd_intra_maternal, ins_maternal, del_maternal,
        #                                                                  set(rm_ind_intra_m), set(rm_ind_ins_m), set(rm_ind_del_m))
        # intra_filt_df_p, ins_filt_df_p, del_filt_df_p = process_used_ind(bnd_intra_paternal, ins_paternal, del_paternal,
        #                                                                  set(rm_ind_intra_p), set(rm_ind_ins_p), set(rm_ind_del_p))

        """ Variant integration (dictionary-based) """
        # First, get reference fasta file
        # t_1 = time.time()
        fa = load_fasta(path_fa_dir, chrom)
        dict_fa = as_dict(fa, chrom)

        if ht == 'm':       # if maternal
            print('Processing variants for %s for maternal genome' % chrom)
            fa_dict_updated_m = process_intrachr_dict(dict_fa, coord_intra_filt_m)
            fa_dict_updated_m = process_dups_dict(fa_dict_updated_m, coord_dups_m)
            fa_dict_updated_m = process_ins_dict(fa_dict_updated_m, coord_ins_filt_m)
            fa_dict_updated_m = process_del_dict(fa_dict_updated_m, coord_del_filt_m)
            fa_dict_updated_m = process_snps_dict(fa_dict_updated_m, coord_snps_m)
            fa_str_final_m = dict_bases_to_str(fa_dict_updated_m)
            fa_chr_dict(fa_str_final_m, outdir_fasta, ht, chrom)
        elif ht == 'p':     # if paternal
            print('Processing variants for %s for paternal genome' % chrom)
            fa_dict_updated_p = process_intrachr_dict(dict_fa, coord_intra_filt_p)
            fa_dict_updated_p = process_dups_dict(fa_dict_updated_p, coord_dups_p)
            fa_dict_updated_p = process_ins_dict(fa_dict_updated_p, coord_ins_filt_p)
            fa_dict_updated_p = process_del_dict(fa_dict_updated_p, coord_del_filt_p)
            fa_dict_updated_p = process_snps_dict(fa_dict_updated_p, coord_snps_p)
            fa_str_final_p = dict_bases_to_str(fa_dict_updated_p)
            fa_chr_dict(fa_str_final_p, outdir_fasta, ht, chrom)


def run_functions_dict_test(ht):
    vcf_dir = '/Users/jayluo/Research_all/Snyder_Lab/vcf_ENCODE/test_custom/'
    # f1, f2, f3, f4 = 'snps_art_eg1.vcf', 'dup_art_eg1.vcf', 'indels_art_eg1.vcf', 'bnd_art_eg1.vcf'
    # f1, f2, f3, f4 = 'dup_art_eg1.vcf', 'snps_art_eg1.vcf', 'indels_art_eg2.vcf', 'bnd_art_eg2.vcf'
    f1, f2, f3, f4 = 'dup_art_eg1.vcf', 'snps_art_incl_unphased.vcf', 'indels_art_eg1.vcf', 'bnd_art_eg1.vcf'
    paths = [vcf_dir + f1, vcf_dir + f2, vcf_dir + f3, vcf_dir + f4]
    cols = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'K562']
    outdir_fasta = '/Users/jayluo/Research_all/Snyder_Lab/vcf_ENCODE/test_custom/dict_program_output/incl_unphased/'


    chrom = 'chr1'

    var_df_all = load_vcf(paths, chrom, cols)

    snps_only = vcf_filt(var_df_all, 'SNP', chrom)
    ins_only = vcf_filt(var_df_all, 'INS', chrom)
    del_only = vcf_filt(var_df_all, 'DEL', chrom)
    dup_only = vcf_filt(var_df_all, 'DUP', chrom)
    inv_only = vcf_filt(var_df_all, 'INV', chrom)       # TODO: check why it's not used (error?) [done]
    bnd_only = vcf_filt(var_df_all, 'BND', chrom)
    bnd_intrachromosomal = filt_bnd(bnd_only, chrom)[0]
    bnd_interchromosomal = filt_bnd(bnd_only, chrom)[1]

    # Add '<INV>' alleles to bnd_intrachromosomal
    bnd_intrachromosomal = pd.concat([bnd_intrachromosomal, inv_only]).reset_index(drop=True)

    # Get maternal and paternal dataframes
    bnd_inter_maternal, bnd_inter_paternal = bnd_subset_gt(bnd_interchromosomal)
    bnd_intra_maternal, bnd_intra_paternal = bnd_subset_gt(bnd_intrachromosomal)

    ins_mult_alt_m, ins_mult_alt_p = mult_alt_df_gt(ins_only)   # still containing some SNPs
    snps_mult_alt_m = vcf_filt(ins_mult_alt_m, 'SNP', chrom)
    snps_mult_alt_p = vcf_filt(ins_mult_alt_p, 'SNP', chrom)

    ins_maternal, ins_paternal = filt_gt(ins_only)
    ins_only_mult_alt_m, ins_only_mult_alt_p = vcf_filt(ins_mult_alt_m, 'INS', chrom), vcf_filt(ins_mult_alt_p, 'INS', chrom)
    ins_merged_m, ins_merged_p = merge_df(ins_maternal, ins_only_mult_alt_m), merge_df(ins_paternal, ins_only_mult_alt_p)

    # Deletions
    del_maternal, del_paternal = filt_gt(del_only)

    # SNPs
    snps_maternal, snps_paternal = filt_gt(snps_only)
    snps_merged_m, snps_merged_p = merge_df(snps_maternal, snps_mult_alt_m), merge_df(snps_paternal, snps_mult_alt_p)

    # Duplications
    dups_maternal, dups_paternal = filt_gt(dup_only)

    # Get coordinates for maternal and paternal dataframes
    coord_inter_m, coord_inter_p = ind_bnd_inter(bnd_inter_maternal), ind_bnd_inter(bnd_inter_paternal)
    coord_intra_m, coord_intra_p = ind_bnd_intra(bnd_intra_maternal), ind_bnd_intra(bnd_intra_paternal)
    coord_ins_m, coord_ins_p = ind_ins(ins_merged_m), ind_ins(ins_merged_p)
    coord_del_m, coord_del_p = ind_del(del_maternal), ind_del(del_paternal)
    coord_snps_m, coord_snps_p = ind_snps(snps_merged_m), ind_snps(snps_merged_p)
    coord_dups_m, coord_dups_p = ind_dups(dups_maternal), ind_dups(dups_paternal)

    # Get filtered coordinates
    coord_intra_filt_m, rm_ind_intra_m = coor_rm_intra(coord_inter_m, coord_intra_m)
    coord_intra_filt_p, rm_ind_intra_p = coor_rm_intra(coord_inter_p, coord_intra_p)

    coord_ins_filt_m, rm_ind_ins_m = coor_rm_ins(coord_intra_filt_m, coord_ins_m)
    coord_ins_filt_p, rm_ind_ins_p = coor_rm_ins(coord_intra_filt_p, coord_ins_p)

    coord_del_filt_m, rm_ind_del_m = coor_rm_del(coord_inter_m, coord_ins_filt_m, coord_del_m)
    coord_del_filt_p, rm_ind_del_p = coor_rm_del(coord_inter_p, coord_ins_filt_p, coord_del_p)

    # Following block necessary?
    # intra_filt_df_m, ins_filt_df_m, del_filt_df_m = process_used_ind(bnd_intra_maternal, ins_maternal, del_maternal,
    #                                                                  set(rm_ind_intra_m), set(rm_ind_ins_m), set(rm_ind_del_m))
    # intra_filt_df_p, ins_filt_df_p, del_filt_df_p = process_used_ind(bnd_intra_paternal, ins_paternal, del_paternal,
    #                                                                  set(rm_ind_intra_p), set(rm_ind_ins_p), set(rm_ind_del_p))

    """ Variant integration (dictionary-based) """
    # First, get reference fasta file
    # t_1 = time.time()
    fa = load_fasta('/Users/jayluo/Research_all/Snyder_Lab/vcf_ENCODE/test_custom/', chrom)
    dict_fa = as_dict(fa, chrom)

    # Maternal

    # TODO: function to process 'N' bases (randomly select from A, T, C, G?) needed?
    if ht == 'm':
        fa_dict_updated_m = process_intrachr_dict(dict_fa, coord_intra_filt_m)
        fa_dict_updated_m = process_dups_dict(fa_dict_updated_m, coord_dups_m)
        fa_dict_updated_m = process_ins_dict(fa_dict_updated_m, coord_ins_filt_m)
        fa_dict_updated_m = process_del_dict(fa_dict_updated_m, coord_del_filt_m)
        fa_dict_updated_m = process_snps_dict(fa_dict_updated_m, coord_snps_m)
        fa_str_final_m = dict_bases_to_str(fa_dict_updated_m)
        fa_chr_dict(fa_str_final_m, outdir_fasta, ht, chrom)
    elif ht == 'p':
        fa_dict_updated_p = process_intrachr_dict(dict_fa, coord_intra_filt_p)
        fa_dict_updated_p = process_dups_dict(fa_dict_updated_p, coord_dups_p)
        fa_dict_updated_p = process_ins_dict(fa_dict_updated_p, coord_ins_filt_p)
        fa_dict_updated_p = process_del_dict(fa_dict_updated_p, coord_del_filt_p)
        fa_dict_updated_p = process_snps_dict(fa_dict_updated_p, coord_snps_p)
        fa_str_final_p = dict_bases_to_str(fa_dict_updated_p)
        fa_chr_dict(fa_str_final_p, outdir_fasta, ht, chrom)

        # print('%s (%s) processing time (incl. fasta generation and'
        #       ' conversion to final string): %.3f seconds' % (chrom, ht, time.time() - t_1))

    # return fa_str_final
