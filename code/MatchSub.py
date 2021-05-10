import sys
import os
# from timeit import repeat
import time
import numpy as np
# import numba

ACGU_NUM={
    'A':0,
    'C':1,
    'G':2,
    'U':3,
}

MATCH_BOX={
    'A':{'A':0, 'C':0, 'G':0, 'U':1},
    'C':{'A':0, 'C':0, 'G':1, 'U':0},
    'G':{'A':0, 'C':1, 'G':0, 'U':1},
    'U':{'A':1, 'C':0, 'G':1, 'U':0},
}

def Match(seq, m0, m1, B):
    for i in range(-B, B+1):
        if MATCH_BOX[seq[m0+i]][seq[m1-i]]==0:
            return False
    return True

def CountMatch(seq, W):
    L=len(seq)
    B=W//2
    cnt = np.zeros([L], dtype=float)
    for i in range(B, L-B):
        for j in range(i+W, L-B):
            if Match(seq, i, j, B):
                # print('Match:', seq[(i-B):(i+B+1)], seq[(j-B):(j+B+1)])
                cnt[i]+=1./L
                cnt[j]+=1./L
    return cnt

if __name__ == '__main__':
    seq = 'AACGCGAAGAACCUUACCUGGGCUUGACAUGCACAGGAAACGCUCGUGAAAGCGAGCGCCUCCCGCAAGGGAUCUGUGCACAGGUGGUGCAUGGCUGUCGUCAGCUCGUGCCGUGAGGUGUUGGGUUAAGUCCCGCAACGAGCGCAACCCCUGUCCUUAGUUGAAUCUUCUAGGGAGACUGCCGGGCGAAACCCGGAGGAAGGUGGGGAUGACGUCAAGUCCGCAUGCCCUUUAUGUCCAGGGCUACACACACGCUACAAUGGCCGGUACAACGGGUUCCGACACGGCGACGUGAAGGCAAUCCCUUAAAGCCGGUCUCAGUUCGGAUUGUUGGCUGCAACUCGCCAGCAUGAAGUCGGAGUUGCUAGUAACCGCAGGUCAGCACACUGCGGUGAAUACGUUCCCGGGCCUUGUACACACCGUAGUAC'
    print('main:')
    start_time = time.time()
    for i in range(100):
        cnt = CountMatch(seq, W=3)
    print('time =', time.time()-start_time)
    print(np.sum(cnt))



