import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

data = pd.read_csv('creditcard.csv')

data_majority = data[data['Class'] == 0]
data_minority = data[data['Class'] == 1]

data_minority_upsampled = resample(data_minority, replace=True, n_samples=282652, random_state=99)
data_final = pd.concat([data_minority_upsampled,data_majority])

class_counts = data_final['Class'].value_counts()

plt.bar(class_counts.index, class_counts.values, color=['blue', 'red'])

plt.title('Sınıf Dağılımı')
plt.xlabel('Sınıf')
plt.ylabel('Veri Sayısı')
plt.xticks(class_counts.index, ['Minority', 'Majority'])


plt.show(block = False)

input("Devam Etmek için Enter tuşuna basın.")
