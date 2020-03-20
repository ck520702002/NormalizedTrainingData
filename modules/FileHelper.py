import re
import collections

class FileHelper():
    
    def __init__(self, fileName):

        self.fileName = fileName
        self.keyList = collections.defaultdict(dict)
        self.fileList = collections.defaultdict(list)
        self.jvalueList = {}
    
    def handleFile(self):
        with open(self.getFileName()) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if len(lines[i]) != 0:
                    key, schemaList, jvalue = self.extractFileContent(lines[i])
                    self.keyList[i] = key
                    self.fileList[i] = schemaList
                    self.jvalueList[i] = jvalue         

    def handleAcyclicFile(self):
         with open(self.getFileName()) as f:
            lines = f.readlines()
            count=0
            tempKey=""
            for i in range(len(lines)):
                if i == 0:
                    continue    # header               
                # 儲存key
                leftBracket = lines[i].find('{')
                rightBracket = lines[i].rfind('}')
                t1 = lines[i][leftBracket+1:rightBracket]
                if lines[i].startswith('"separator'):
                    count+=1
                    tempKey=t1
                    self.keyList[count][tempKey]=list()           
                elif lines[i].startswith('cluster'):
                    self.fileList[count].append(t1)
                    #tempCluster=self.removeKey(tempKey, t1)
                    self.keyList[count][tempKey].append(t1)
                elif lines[i].startswith('separator'):  
                    tempKey=t1
                    self.keyList[count][tempKey]=list()

    def removeKey(self, key, cluster):
        keyList = key.replace(" ", "").split(",")
        keySet = set(keyList)
        clusterList = cluster.replace(" ", "").split(",")

        string=""

        for c in clusterList:
            if not c in keySet:
                string+=c
                string+=","
        
        clusterRes = string[:-1].replace(" ", "").split(",")
        return clusterRes         

    def extractFileContent(self, content):

        leftBracket = content.find('{')
        rightBracket = content.rfind('}')
        t1 = content[leftBracket+1:rightBracket]
        print(t1)
        t1List = t1.split('|')

        pattern = "{(.*?)}"

        key = re.findall(pattern, t1List[0], re.DOTALL)
        val = re.findall(pattern, t1List[1], re.DOTALL)
        return key,val,content[rightBracket+2:]


    def getFileName(self):
        return self.fileName

    def getKeyList(self):
        return self.keyList

    def getFileList(self):
        return self.fileList

    def getJvalueList(self, line):
        return self.jvalueList[line]    


    def handleAcyclicScheme(self):
        with open(self.getFileName()) as f:
            lines = f.readlines()
            for line in lines:
                if len(line) != 0:
                    contents = line.split(':')
                    level = contents[1].replace(" ","")
                    leftBracket = contents[2].find('{')
                    rightBracket = contents[2].rfind('}')
                    t1 = contents[2][leftBracket+1:rightBracket]                                                                      
                    if line.startswith("separator"): #key
                        self.keyList[int(level)] = t1
                    elif line.startswith("cluster"): #table
                        self.fileList[int(level)-1].append(t1)





