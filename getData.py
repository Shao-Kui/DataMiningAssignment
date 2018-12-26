import csv
import numpy as np

# attribute=['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id',
#            'admission_source_id', 'time_in_hospital', 'payer_code', 'medical_specialty', 'num_lab_procedures', 'num_procedures',
#            'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'max_glu_serum',
# 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
# 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin',
# 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
# 'diabetesMed', 'readmitted']
attribute=['diag_1', 'diag_2', 'diag_3', 'readmitted']
dic={}
artTonum={}
csvfile=open(r'diabetic_data_diag.csv')
readCSV = csv.DictReader(csvfile, delimiter=',')

#for row in readCSV:
 #   for key in row.keys():
 #       attribute.append(key)
 #   break

print("attribute:",attribute)

for art in attribute:
    dic[art]={}
    artTonum[art]=0

print(dic)
for row in readCSV:
    #print(row.keys())
    #print(len(row))
    #print(row)
    #for item in row:
    #    print(row[item])
    for key in row.keys():
        value=row[key]
        if  value not in dic[key]:
            if value != "?":
                #dic[key][value] = -1
                dic[key][value]=artTonum[key]
                artTonum[key]=artTonum[key]+1

print(artTonum)
for key in dic.keys():
    print(key,":",dic[key])

totalAttribute=0
for key in artTonum.keys():
    if key != "encounter_id" and key != "patient_nbr" and key != "readmitted":
        totalAttribute = totalAttribute+artTonum[key]

csvfile.seek(0,0)
readCSV1 = csv.DictReader(csvfile, delimiter=',')

resultList=[]
for row in readCSV1:
    for key in row.keys():
        if key!="encounter_id" and key!="patient_nbr" and key!="readmitted":
            value=row[key]
            onepoint=0
            if value == "?":
                onepoint=-1
            else:
                onepoint=dic[key][value]
            i=0
            length=artTonum[key]
            while i<length:
                if i == onepoint:
                    resultList.append(1)
                else:
                    resultList.append(0)
                i+=1

resultTmp=np.asarray(resultList)

result=resultTmp.reshape(int(len(resultList)/totalAttribute),totalAttribute)
#resultF=np.floor(result)
print(result[1])
print(result.shape)
#np.savetxt("result.txt",result)
np.save("attribute_diag",result)
#print(result)


csvfile.seek(0,0)
readCSV2 = csv.DictReader(csvfile, delimiter=',')

lableList=[]
for row in readCSV2:
    for key in row.keys():
        if key=="readmitted":
            value=row[key]
            onepoint=0
            if value == "?":
                onepoint=-1
            else:
                onepoint=dic[key][value]
            i=0
            length=artTonum[key]
            while i<length:
                if i == onepoint:
                    lableList.append(1)
                else:
                    lableList.append(0)
                i+=1

lableTmp=np.asarray(lableList)
lableResult=lableTmp.reshape(int(len(lableList)/3),3)
#resultF=np.floor(result)
print(lableResult[18])
print(lableResult.shape)
#np.savetxt("result.txt",result)
# np.save("lable",lableResult)

print("abcc")

#hxq=np.load("arrtribute.npy")
#print("cdee")




csvfile.close()
