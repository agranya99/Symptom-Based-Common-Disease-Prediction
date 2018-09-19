from sklearn.externals import joblib
import numpy as np
import pandas as pd

clf = joblib.load('RandomForestF.pkl')
df = pd.read_csv('Final_DataFrame.csv', encoding = 'utf-8')
col = df.columns[1:]
xd = 1
while(not xd == 0):
    test_text = input("Symptoms separated by ;  ")
    test_symptoms = test_text.split(';')
    sym_list = list(col)
    test_data = np.zeros(len(sym_list))
    flag = 0
    for i in test_symptoms:
        #print(i)
        try:
            ind = sym_list.index(" ".join(i.strip(' ').split()))
            #print(ind)
            test_data[ind] = 1
            flag += 1
        except ValueError:
            pass
    tdf = pd.DataFrame([test_data], columns = sym_list).astype(int)
    #print(tdf)
    #print(flag)
    #print(test_data)
    print(flag, "value(s) accepted")
    #print(clf.predict(tdf))
    out = clf.predict_proba(tdf)
    out1 = out.tolist()
    out_list = out1[0]
    #print("XX", len(out_list))
    dis_list = list(df['Disease'])
    print("Probable Diseases:\t% Probability\n\n")
    for i in range(len(out_list)):
            if not out_list[i] == 0:
                #print(i)
                print(dis_list[i], "\t", ((out_list[i]) * 100))
                #print(out_list)
    stopword = input("press X to stop")
    if (stopword == 'x' or stopword == 'X'):
        xd = 0
