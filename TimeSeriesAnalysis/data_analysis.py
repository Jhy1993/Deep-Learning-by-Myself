#import matplotlib.pyplot as plt
import numpy as np
import xlrd
import xlwt

def sort(filename):
    data = xlrd.open_workbook(filename)
    table = data.sheets()[0]
    ncols = table.ncols
    nrows = table.nrows
    ndata = []
    for i in range(1,ncols):
        l = sortlist(table.col_values(i))
        ndata.append(l)
    ndata = np.array(ndata).transpose()
    wb = xlwt.Workbook()
    sheet = wb.add_sheet("ranks")
    for i in range(nrows):
	print i
        for j in range(ncols):
            if j == 0:
                v = table.cell(i,0).value
            else:
                v = ndata[i][j-1]
            sheet.write(i,j,v)
    wb.save("new_rank.xls")
    return 1
def sortlist(datalist):
    newdata = datalist
    rg = len(datalist)
    rdata = [0] * rg
    ids = [0] * rg
    for i in range(rg):
        ids[i] = i
    for i in range(rg):
        for j in range(i,rg):
            if newdata[i] > newdata[j]:
                newdata[i],newdata[j] == newdata[j],newdata[i]
                ids[i],ids[j] = ids[j],ids[i]
    cn = 0
    for i in range(rg):
        location = ids[i]
        if newdata[i] == 0:
            rdata[location] = -1
            cn += 1
        elif newdata[i] > 0:
            rdata[location] = i - cn
    #print datalist
    return rdata
  
if __name__ == "__main__":
    filename = "analysis_id.txt"
    sort("data.xlsx")
