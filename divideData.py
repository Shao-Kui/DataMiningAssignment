import numpy as np

attribute=np.load("attribute.npy")
attribute_diag=np.load("attribute_diag.npy")
lable=np.load("lable.npy")

permutation = list(np.random.permutation(len(attribute)))
print(len(permutation))

newAttribute=attribute[permutation,:]
newAttribute_diag=attribute_diag[permutation,:]
newLable=lable[permutation,:]

i=10000
attributeList = np.vsplit(newAttribute, [i,2*i,3*i,4*i,5*i,6*i,7*i,8*i,9*i])
attribute_diagList = np.vsplit(newAttribute_diag, [i,2*i,3*i,4*i,5*i,6*i,7*i,8*i,9*i])
lableList = np.vsplit(newLable,[i,2*i,3*i,4*i,5*i,6*i,7*i,8*i,9*i])

k=0
while k<10:
    j=0
    trainAttrSet=[]
    trainAttr_diagSet=[]
    trainLabSet=[]
    while j<10:
        if j!=k:
            trainAttrSet.append(attributeList[j])
            trainAttr_diagSet.append(attribute_diagList[j])
            trainLabSet.append(lableList[j])
        j+=1

    trainAttr=np.vstack(trainAttrSet)
    trainAttr_diag = np.vstack(trainAttr_diagSet)
    trainLab=np.vstack(trainLabSet)


    testAttr=attributeList[k]
    testAttr_diag = attribute_diagList[k]
    testLab=lableList[k]

    np.save("trainAttribute"+str(k),trainAttr)
    np.save("trainAttribute_diag"+str(k),trainAttr_diag)
    np.save("trainLable"+str(k),trainLab)
    np.save("testAttribute"+str(k),testAttr)
    np.save("testAttribute_diag" + str(k), testAttr_diag)
    np.save("testLable"+str(k),testLab)

    k+=1



