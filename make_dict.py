from pandas import *
import json

# Handful of calibration points, use the far ends
# U layer 1 and 5 are measured
# Start from calibration, use closest one
# Calculate the rest by interpolation

xlsx = ExcelFile('TensionGcodeOye.xlsx')
for sheetind in range(len(xlsx.sheet_names)):
    df = xlsx.parse(xlsx.sheet_names[sheetind])
    sheetstr = xlsx.sheet_names[sheetind]
    print(sheetstr)
    Wire = df.Wire.values.tolist()
    X = df.iloc[:, [8]]
    Y = df.iloc[:, [9]]

    Ylist = Y.values.tolist()
    Xlist = X.values.tolist()

    gcode = []
    for i in range(len(Wire)):
        gcode.append(str(Xlist[i][0])+" "+str(Ylist[i][0]))

    wiredict = {Wire[i]: gcode[i] for i in range(len(Wire))}
    with open(sheetstr+".json", "w") as outfile:
        json.dump(wiredict, outfile)
