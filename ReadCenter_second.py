import json
import numpy as np
survey_json = "C:\\Users\\dell\\Desktop\\中科院\\survey56\\survey56.json"
center_csv = "C:\\Users\\dell\\Desktop\\中科院\\survey56\\survey56centers\\cts_8"
with open(survey_json, 'r', encoding='utf8') as fp:
    qsfile = json.load(fp)
centroids = np.genfromtxt(center_csv, dtype='int', delimiter=',').tolist()
with open("survey56centers/cts_names", 'r') as fp:
    qids = fp.readlines()[0].strip().split(',')
qids = [qid[3:].strip().split('_') for qid in qids]
qids_0=[]
for i in qids:
    qids_0.append(i[0])
qidsList = [qs["qid"] for qs in qsfile["questions"]]
center_qids = []
num=[]
for i in set(qids_0):
      num.append([i,qids_0.count(i)])
for qsidx, qs in enumerate(qidsList):
    for i in range(len(qids_0)):
        if qs == qids_0[i]:
            center_qids.append(qsfile["questions"][qsidx])   
answers = [','.join([qs["body"] for qs in center_qids])]
print(num)
print(len(center_qids))
#人工操作
a=[0,0,0,0,8,-1,-1,-1,-1,-1,-1,-1,6,-1,-1,-1,-1,-1,6,-1,-1,-1,-1,-1,6,-1,-1,-1,-1,-1,5,-1,-1,-1,-1,6,-1,-1,-1,-1,-1,4,-1,-1,-1,7,-1,
   -1,-1,-1,-1,-1,6,-1,-1,-1,-1,-1,6,-1,-1,-1,-1,-1,5,-1,-1,-1,-1,7,-1,-1,-1,-1,-1,-1,6,-1,-1,-1,-1,-1,7,-1,-1,-1,-1,-1,-1,4,-1,-1,-1,5,-1,
   -1,-1,-1,0]
t=[a,a,a,a,a,a,a,a]
for center,qs in zip(centroids,t):
    answer = []
    k=0
    for idx in range(len(center)):
        if (qs[idx] >= 1):
            for j in range(qs[idx]):
                h=idx+j
                if (center[h] == 0):
                    ct_ans=' '
                    answer.append(ct_ans)
                else:
                    ct_ans = center_qids[k]["option_list"][j]["description"]
                    answer.append(ct_ans)
                k+=1
        elif (qs[idx] == 0):
            ct_ans = center_qids[k]["option_list"][center[idx]]["description"]
            answer.append(ct_ans)
            k+=1
        else:
            continue
    answer_txt = ','.join(answer)
    answers.append(answer_txt)
with open('survey56report\\trans.csv', 'w') as fp:
    for line in answers:
        fp.write(line)
        fp.write('\n')
