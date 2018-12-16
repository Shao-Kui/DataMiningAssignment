import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import csv

reader = csv.reader(open("diabetic_data.csv", "rt"), delimiter=",")
result = list(reader)
result = pd.DataFrame(result, dtype="category")
print("Start Imputation")
imp = SimpleImputer(missing_values="?", strategy="most_frequent")
result = imp.fit_transform(result)
print("End Imputation")
result = np.array(result, dtype="<U36")
np.savetxt("complete.csv", result, delimiter=",", fmt="%s")
