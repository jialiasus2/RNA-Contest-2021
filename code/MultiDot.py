
import sys 
sys.path.append('/home/aistudio/external-libraries')

import os
import numba
import numpy as np
import pahelix.toolkit.linear_rna as linear_rna

# @numba.jit(nopython=True)
def Match(x, y):
    if (x=='A' and y=='U') or (x=='C' and y=='G') or (x=='G' and (y=='C' or y=='U')) or(x=='U' and (y=='A' or y=='G')):
        return True
    else:
        return False

# @numba.jit(nopython=True)
def GetMultiDot(seq, dot_cnt=3, sel=3):
    dots=[]
    l=len(seq)
    print('len =', l)
    for _ in range(dot_cnt):
        dot = list('?'*l)
        i=0
        j=l-1
        c=0
        while i<j and c<sel:
            if np.random.uniform(0., 1.)<0.5:
                i+=1
                continue
            k=seq[i]
            while j>i and not Match(k, seq[j]):
                j-=1
            if j>i:
                dot[i]='('
                dot[j]=')'
                c+=1
            if c==sel or i>=j:
                break
            i+=1
            j-=1
        dot = ''.join(dot)
        print(dot)
        dot = linear_rna.linear_fold_c(seq, use_constraints=True, constraint=dot)[0]
        print(dot)
        dots.append(dot)
    return dots
# seq = train_datas[0][1]
seq = 'AACGCGAAGAACCUUACCUGGGCUUGACAUGCACAGGAAACGCUCGUGAAAGCGAGCGCCUCCCGCAAGGGAUCUGUGCACAGGUGGUGCAUGGCUGUCGUCAGCUCGUGCCGUGAGGUGUUGGGUUAAGUCCCGCAACGAGCGCAACCCCUGUCCUUAGUUGAAUCUUCUAGGGAGACUGCCGGGCGAAACCCGGAGGAAGGUGGGGAUGACGUCAAGUCCGCAUGCCCUUUAUGUCCAGGGCUACACACACGCUACAAUGGCCGGUACAACGGGUUCCGACACGGCGACGUGAAGGCAAUCCCUUAAAGCCGGUCUCAGUUCGGAUUGUUGGCUGCAACUCGCCAGCAUGAAGUCGGAGUUGCUAGUAACCGCAGGUCAGCACACUGCGGUGAAUACGUUCCCGGGCCUUGUACACACCGUAGUAC'
dots = GetMultiDot(seq)
print(dots)
print(linear_rna.linear_fold_c(seq))