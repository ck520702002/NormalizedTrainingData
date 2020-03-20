from glob import glob

from modules.config import config
from modules.MyDatabase import MyDatabase
from modules.FileHelper import FileHelper
import collections


def main():
    # read values from a section
    directory = config.get('schema', 'directory')
    baselineFile = config.get('schema', 'baselineFile')
    baselineCols = config.get('schema', 'baselineCols')
    acyclicDic = config.get('schema', 'acyclicDic')
    holdOutFile = config.get('schema', 'holdoutFile')

    # store columns both in a list and a dictionary
    columns = list()
    colsDict = {}
    with open(baselineCols) as f:
        lines = f.readlines()
        for i in lines:
            cols = i.split(',')
            for index, col in enumerate(cols):
                columns.append(col)
                colsDict[index] = col

    # connect to PostgreSQL
    db = MyDatabase()
    # get baseline tablename
    fileNameList = baselineFile.split('/')
    tempList = fileNameList[-1].split('.')
    tableName = tempList[0].replace("-", "_")
    # insert baseline into database
    db.importBaselineData(baselineFile, tableName, columns)

    holdoutFileList = holdOutFile.split('/')
    htempList = holdoutFileList[-1].split('.')
    htableName = htempList[0].replace("-", "_")
    # insert baseline into database
    db.importBaselineData(holdOutFile, htableName, columns)

    tempFile = FileHelper(acyclicDic)
    tempFile.handleAcyclicFile()

    acyclicFileList = tempFile.getFileList()
    if len(acyclicFileList) != 0:
        for level, schemeList in acyclicFileList.items():
            for scheme in schemeList:
                tempSchema = scheme.replace(" ", "").replace(",", "_")
                schemaTableName = tempList[0] + '_' + str(level) + '_' + tempSchema
                # insert acyclic schemas into database
                db.acyclicSchemaImport(tableName, schemaTableName.replace("-", "_"), scheme, colsDict)
    

    testDic = tempFile.getKeyList();

    for scheme, schemeLevel in  testDic.items():    #scheme number
        tableViewName = tempList[0]+"_v_"+str(scheme) 

        viewMatrix=list()
        for separator, separatorLevel in schemeLevel.items():   #key
            separatorCols=list()
            keyList = list()
            if len(separator) > 1:
                keyList = separator.replace(" ", "").split(",") # '6, 8, 10' => list [6,8,10] 
            elif len(separator) == 1 and separator.isdigit():
                keyList = [separator]                           # '6' => 

            selectString = "select "
            fromString = " from "
            for i in range(len(separatorLevel)):  #cluster
                if i == 0: # firstTable should add on key columns

                    if len(keyList) > 0:
                        for key in keyList:
                            selectString += "t"+str(i)+"."+colsDict[int(key)]+","
                            separatorCols.append(colsDict[int(key)])
                restColList = tempFile.removeKey(separator, separatorLevel[i])
                if len(restColList) > 0:
                    for restCol in restColList:
                        selectString += "t"+str(i)+"."+colsDict[int(restCol)]+"," 
                        separatorCols.append(colsDict[int(restCol)])
                tempSchema = separatorLevel[i].replace(" ", "").replace(",", "_")
                tablename = tempList[0] + "_" + str(scheme) + "_" + tempSchema
                if i == 0:
                    fromString += tablename + " " + "t"+str(i) + " "
                else:
                    if len(keyList) == 0:
                        fromString += " cross join "+tablename + " " + "t"+str(i) + " " 
                    else:                      
                        fromString += " join "+tablename + " " + "t"+str(i) + " on " 

                    for j in range(len(keyList)):
                        fromString += "t"+str(i)+"."+colsDict[int(keyList[j])]+"="+"t"+str(i-1)+"."+colsDict[int(keyList[j])]+" "
                        if j< len(keyList)-1:
                            fromString += "and "
            # finish first layer... 
            viewMatrix.append([separatorCols, selectString[:-1]+fromString])               
            #print(sqlString+selectString[:-1]+fromString+") as v1;") 
        

        queue = [viewMatrix[0]]
        lCount = 0
        rCount = 1
        bigSelectSql=""
        for k in range(1, len(viewMatrix)):

            viewSelect = list()
            crossJoin = True
            innerSelectString= " select " 
            innerFromString = " from "
            leftCols, leftSql = queue.pop()


            colDict=dict()
            for col in viewMatrix[k][0]:
                if col not in colDict:
                    colDict[col] = 1
                else:
                    colDict[col] += 1
            for col in leftCols:
                if col not in colDict:
                    colDict[col] = 1
                else:
                    colDict[col] += 1 

            for left in leftCols:
                innerSelectString += "v"+str(lCount)+"."+left+"," 
                viewSelect.append(left)

            for right in viewMatrix[k][0]:
                if colDict[right] == 1:    # exclude keys
                    innerSelectString += "v"+str(rCount)+"."+right+"," 
                    viewSelect.append(right)
                elif colDict[right] > 1:
                    crossJoin = False
            innerFromString += "(" + leftSql +") as v" + str(lCount) 
            if not crossJoin:
                innerFromString +=  " join (" + viewMatrix[k][1] + ") as v" + str(rCount) 
            else:
                innerFromString +=  " cross join (" + viewMatrix[k][1] + ") as v" + str(rCount)     

            keyCount=0
            for col, colv in colDict.items():
                if colv > 1:
                    if keyCount == 0:
                        innerFromString += " on "
                    else:
                        innerFromString += " and "
                    innerFromString += " v"+str(lCount)+"."+col+"=v"+str(rCount)+"."+col
                    keyCount+=1
            queue.append([viewSelect, innerSelectString[:-1]+innerFromString])
                    

        if len(queue) == 1:
            finalCols, finalSql = queue.pop()
            deletefinalString = "drop table if exists "+tableViewName+";"
            finalString = "select * into "+tableViewName+" from (" + finalSql + ") as vt;"
            #print(finalString)
            db.query(deletefinalString)
            db.query(finalString)               

    # parse sep files
    '''
    schemaMap = {}
    fileMapList = glob(directory)
    for i in fileMapList:
        tempFile = FileHelper(i)
        tempFile.handleFile()
        fileNmaeList = i.split('.')     # fileName ex: airports.csv.TO.60.RANGE.9.THRESH.0.0.sep, get 0 0 here...
        
        keyList = tempFile.getKeyList() 

        if len(keyList) != 0:
            for line, key in keyList.items():
                tableNameList=[]
                schemaList = tempFile.getFileList(line)
                for schema in schemaList:
                    tempSchema = schema.replace(" ", "").replace(",", "")
                    schemaTableName = tempList[0] + '_'+ fileNmaeList[-3] + '_' + fileNmaeList[-2] + '_' + str(line) +'_' + tempSchema
                    # insert acyclic schemas into database
                    db.acyclicSchemaImport(tableName, schemaTableName, key, schema, colsDict)
                    tableNameList.append(schemaTableName)       
                viewName = 'view_' + tempList[0] + '_'+ fileNmaeList[-3] + '_' + fileNmaeList[-2] + line
                db.createView(viewName, tableNameList, tempFile.getJvalueList(line), key, colsDict)
    '''
    #db.close()


if __name__ == "__main__":
    main()
