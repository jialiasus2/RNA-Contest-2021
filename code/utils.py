
# 训练集文件
TRAIN_TXT = './data/data68535/train.txt'
# 验证集文件
DEV_TXT = './data/data68535/dev.txt'
### 测试集A
# TEST_TXT = './data/data68535/test_nolabel.txt'
# 测试集B
TEST_TXT = 'data/data82504/B_board_112_seqs.txt'

# 所有文件（包括训练、验证和测试）中的RNA合并写为fasta格式的存储路径
SEQ_FASTA = './work/features/all.fasta'

# contrafold提取的结构特征，四个文件分别是在不同参数下生成的
CONTRAFOLD_FEATURE_FILES = [
    './work/features/contrafold_default.patents',
    './work/features/contrafold_viterbi.patents',
    './work/features/contrafold_noncomp.patents',
    './work/features/contrafold_noncomp_viterbi.patents',
]

# LinearFold 在foldc下生成的特征文件
FOLDC_FEATURE_FILE = './work/features/foldc.txt'

def ReadLines(path):
    with open(path, 'r') as f:
        return [l.strip() for l in f.readlines() if len(l.strip())>0]
def WriteLines(path, lines):
    with open(path, 'w') as f:
        f.writelines([l+'\n' for l in lines])

def List2Dict(datas):
    out = {}
    for data in datas:
        out[data[0]] = data
    return out

def WriteFasta(path, datas):
    lines = []
    for data in datas:
        lines.append(data[0])
        lines.append(data[1])
        if len(data)>2:
            lines.append('>structure')
            lines.append(data[2])
    WriteLines(path, lines)

def ReadFasta(path, with_structure=True):
    lines = ReadLines(path)
    L = len(lines)
    step = 4 if with_structure else 2
    datas = []
    for i in range(0, L, step):
        if with_structure:
            datas.append((lines[i], lines[i+1], lines[i+3]))
        else:
            datas.append((lines[i], lines[i+1]))
    return datas

def ReadData(path, with_label=True):
    lines = ReadLines(path)
    data = []
    L = len(lines)
    # print(L)
    # print(lines[:10])
    i=0
    while i<L:
        data_id = lines[i]
        # print('read:', data_id)
        i+=1
        seq = lines[i]
        i+=1
        structrue = lines[i]
        i+=1
        if with_label:
            prob = [float(lines[j].split()[1]) for j in range(i, i+len(seq))]
            i+=len(seq)
            data.append((data_id, seq, structrue, prob))
        else:
            data.append((data_id, seq, structrue))
        while i<L and len(lines[i])==0:
            i+=1
    return data
