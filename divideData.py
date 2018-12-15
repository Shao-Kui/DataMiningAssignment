import numpy as np

attribute=np.load("attribute.npy")
lable=np.load("lable.npy")

permutation = list(np.random.permutation(len(attribute)))
print(len(permutation))

newAttribute=attribute[permutation,:]
newLable=lable[permutation,:]

i=10000
attributeList = np.vsplit(newAttribute, [i,2*i,3*i,4*i,5*i,6*i,7*i,8*i,9*i])
lableList = np.vsplit(newLable,[i,2*i,3*i,4*i,5*i,6*i,7*i,8*i,9*i])

k=0
while k<10:
    j=0
    trainAttrSet=[]
    trainLabSet=[]
    while j<10:
        if j!=k:
            trainAttrSet.append(attributeList[j])
            trainLabSet.append(lableList[j])
        j+=1

    trainAttr=np.vstack(trainAttrSet)
    print(trainAttr.shape)
    trainLab=np.vstack(trainLabSet)
    print(trainLab.shape)
    testAttr=attributeList[k]
    print(testAttr.shape)
    testLab=lableList[k]
    print(testLab.shape)
    np.save("trainAttribute"+str(k),trainAttr)
    np.save("trainLable"+str(k),trainLab)
    np.save("testAttribute"+str(k),testAttr)
    np.save("testLable"+str(k),testLab)

    k+=1



