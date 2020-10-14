import pandas as pd
a = [[1,2],[3,4]]
print()
data = DataFrame(a,index=['x1', 'x2'],columns=['y1', 'y2'])
path= r'C:\Users\Administrator\Desktop\files'
for i in range(0,4):
    for j in range(0,4):
        x = data.to_csv(os.path.join(path,str(i) + '-' + str(j) + '.txt'),sep='\t',header=None,index=None)