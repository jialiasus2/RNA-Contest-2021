import os
import sys 
sys.path.append('/home/aistudio/external-libraries')

from tqdm import tqdm

import pahelix.toolkit.linear_rna as linear_rna

from utils import ReadData, ReadFasta, WriteFasta
from utils import TRAIN_TXT, DEV_TXT, TEST_TXT, SEQ_FASTA, FOLDC_FEATURE_FILE

def MakeContrafold(load_file=SEQ_FASTA, save_fold='./work/features'):
    '''
    - 生成contrafold特征，分别在默认参数、viterbi、noncomplementray、
      viterbi+noncomplementray四种模式下生成。
    - 这里有对contrafold源码进行了适当修改，以支持多行fasta格式文件的输入。
    - 经测试，编译生成的可执行文件有时无法直接移植，换了新环境需要重新编译。
    - 编译方法：在contrafold/src目录下执行make，然后将src目录下生成的
      ./contrafold可执行文件拷贝到contrafold/bin目录下。
    '''
    os.system('./contrafold/bin/contrafold>%s predict %s'%(os.path.join(save_fold, 'contrafold_default.patents'), load_file))
    os.system('./contrafold/bin/contrafold>%s predict --viterbi %s'%(os.path.join(save_fold, 'contrafold_viterbi.patents'), load_file))
    os.system('./contrafold/bin/contrafold>%s predict --noncomplementary %s'%(os.path.join(save_fold, 'contrafold_noncomp.patents'), load_file))
    os.system('./contrafold/bin/contrafold>%s predict --noncomplementary --viterbi %s'%(os.path.join(save_fold, 'contrafold_noncomp_viterbi.patents'), load_file))

def MakeFoldC(load_file=SEQ_FASTA, save_file=FOLDC_FEATURE_FILE):
    '''
    生成foldc特征。
    '''
    datas = ReadFasta(load_file, False)
    write_datas = []
    for sid, seq in tqdm(datas):
        structure = linear_rna.linear_fold_c(seq)[0]
        write_datas.append((sid, seq, structure))
    WriteFasta(save_file, write_datas)
        

def MakeFasta(save_file=SEQ_FASTA):
    '''
    将所有数据写入一个fasta文件，方便后续处理。
    '''
    lines = []
    train_datas = ReadData(TRAIN_TXT, True)
    print(len(train_datas))
    dev_datas = ReadData(DEV_TXT, True)
    print(len(dev_datas))
    test_datas = ReadData(TEST_TXT, False)
    print(len(test_datas))
    for data in train_datas+dev_datas+test_datas:
        sid = data[0]
        seq = data[1]
        lines.append(sid+'\n')
        lines.append(seq+'\n')
    with open(save_file, 'w') as f:
        f.writelines(lines)
    print('write to fasta.')

if __name__ == '__main__':
    MakeFasta()
    MakeFoldC()
    MakeContrafold()

