import numpy as np


class DataSet(object):
    def __init__(self,num, prefix="train"):
        self.xs = np.floor(np.load(prefix + "Attribute"+str(num)+".npy") + 0.5)
        self.diags = np.floor(np.load(prefix + "Attribute_diag"+str(num)+".npy") + 0.5)
        self.labels = np.floor(np.load(prefix + "Lable"+str(num)+".npy") + 0.5)
        self.num_examples = len(self.xs)
        self.point=0#表示完成处理的位置
        print(self.xs)

    def next_batch(self, batch_num):
        end=self.point+batch_num
        if end<=self.num_examples:
            xs = self.xs[self.point:end]
            diags = self.diags[self.point:end]
            labels = self.labels[self.point:end]
            self.point=end
        else:
            end=end-self.num_examples
            xs = np.vstack([self.xs[self.point:],self.xs[0:end]])
            diags = np.vstack([self.diags[self.point:],self.diags[0:end]])
            labels = np.vstack([self.labels[self.point:],self.labels[0:end]])
            self.point = end
        return xs, diags, labels


if __name__ == "__main__":
    print("abcc")
    a = DataSet(0)
    b=a.next_batch(101760)
    print(b)
    print(b[0].shape)
    c=a.next_batch(10)
    print(c[0].shape)

