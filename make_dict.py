from pandas import *
import json

# Handful of calibration points, use the far ends
# U layer 1 and 5 are measured
# Start from calibration, use closest one
# Calculate the rest by interpolation

xlsx = ExcelFile('TensionGcodeOye.xlsx')
for sheetind in range(len(xlsx.sheet_names)):
    df = xlsx.parse(xlsx.sheet_names[sheetind])
    print(df.columns)
    sheetstr = xlsx.sheet_names[sheetind]
    df.to_hdf(sheetstr+".df", key="Misc") 
