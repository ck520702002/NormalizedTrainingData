import psycopg2
import psycopg2.extras

from modules.config import config


class MyDatabase():

    def __init__(self):
        self.conn = psycopg2.connect(database=config.get('database', 'dbname'),
                                     user=config.get('database', 'user'),
                                     host=config.get('database', 'host'),
                                     password=config.get('database', 'password'))
        self.cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        self.schemaTableList = {}

    def getSQLResult(self, sql):
        self.cur.execute(sql)
        return self.cur.fetchall()

    def query(self, query):
        print(query)
        self.cur.execute(query)
        self.conn.commit()

    def importBaselineData(self, baselineFile, tableName, cols):
        f_contents = open(baselineFile, 'r')
        self.createTable(tableName, cols)
        self.cur.copy_from(f_contents, tableName, sep=',', columns=cols)
        self.conn.commit()

    def createTable(self, tableName, cols):
        deleteFirstString = "DROP TABLE IF EXISTS " + tableName + ";"
        createTableString = " CREATE TABLE " + tableName + " ("
        for col in cols:
            createTableString += col
            createTableString += " TEXT, "
        createTableString = createTableString[:-2]
        createTableString += " );"
        self.query(deleteFirstString)
        self.query(createTableString)

    def acyclicSchemaImport(self, tableName, schemaTable, tempSchema, cols):
        if len(tempSchema) == 0 or len(cols) == 0 or schemaTable in self.schemaTableList:
            return
        self.schemaTableList[schemaTable] = schemaTable

        colString = ""
        schemaList = tempSchema.replace(" ", "").split(",")
        for i in schemaList:
            if int(i) in cols:
                colString += cols[int(i)]
                colString += ","
        deleteString = "DROP TABLE IF EXISTS " + schemaTable + ";"
        createString = "CREATE TABLE " + schemaTable + " AS SELECT DISTINCT " + colString[:-1] + " FROM " + tableName + " ;"
        self.query(deleteString)
        self.query(createString)

    def close(self):
        self.cur.close()
        self.conn.close()
        self.schemaTableList.clear()
