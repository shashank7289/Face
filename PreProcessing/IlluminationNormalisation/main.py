'''
Created on Aug 26, 2016

@author: uid38420
'''
import method

# database = "D:/Codes/Python/FacePreprocessing/results/clahe/*.jpg"
# database = 'D:/Codes/Git/AI_Bio_Face/11_OpenCV/Python/demo_with_gui_facial_expression/database/baslerfaces_demo/yOliver/*.jpg'
database = "D:/Codes/Git/For everything else/AI_Bio_Face_db_YaleB/yaleB12/*.jpg"

print("Select preprocessing method:")
print("1: grey + stretch")
print("2: stretch + grey")
print("3: retinex")
print("4: histogram equalization")
print("5: clahe")
print("6: gamma correction")
print("7: clahe + gamma correction")
x = input("Your choice? ")

if x == 1:
    case = "greyStretch"
    method.greyStretch(database, case)
elif x == 2:
    case = "stretchGrey"
    method.stretchGrey(database, case)
elif x == 3:
    case = "retinex with adjust"
    method.retinexWithAdjust(database, case)
elif x == 4:
    case = "histogram equalisation"
    method.histogramEqualization(database, case)
elif x == 5:
    case = "clahe"
    method.clahe(database, case)
elif x == 6:
    case = "gamma correction"
    gamma = 1.5
    method.gammaCorrection(database, case, gamma)
elif x == 7:
    case = "gamma correction + clahe"
    gamma = 1.5
    method.gammaCorrectionClahe(database, case, gamma)
else:
    print ("Invalid choice")
