def caculateF1(groundtruthList,outputList):
    TP=[0,0,0]
    FP=[0,0,0]
    FN=[0,0,0]
    for i in range(len(groundtruthList)):
        if groundtruthList[i] == '0':
            if outputList[i] == '0':
                TP[0]+=1
            elif outputList[i] == '1':
                FN[0]+=1
                FP[1]+=1
            elif outputList[i] == '2':
                FN[0]+=1
                FP[2]+=1
        elif groundtruthList[i] =='1':
            if outputList[i] == '0':
                FN[1]+=1
                FP[0]+=1
            elif outputList[i] == '1':
                TP[1]+=1
            elif outputList[i] == '2':
                FN[1]+=1
                FP[2]+=1
        elif groundtruthList[i] =='2':
            if outputList[i]=='0':
                FN[2]+=1
                FP[0]+=1
            elif outputList[i]=='1':
                FN[2]+=1
                FP[1]+=1
            elif outputList[i]=='2':
                TP[2]+=1

    precision=0

    for i in range(len(TP)):
        precision+=TP[i]/(TP[i]+FP[i])

    precision=precision/len(TP)

    recall=0

    for i in range(len(TP)):
        recall+=TP[i]/(TP[i]+FN[i])

    recall=recall/len(TP)


    F1=2*recall*precision/(recall+precision)

    return precision,recall,F1

gtlist=['0','0','0','1','1','1','2','2','2']
outlist=['0','1','1','0','1','2','0','1','2']

# print(caculateF1(gtlist,outlist))



