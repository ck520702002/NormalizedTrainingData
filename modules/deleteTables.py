from glob import glob

from modules.config import config
from modules.MyDatabase import MyDatabase
from modules.FileHelper import FileHelper


def main():
	tableName = input ("Enter table like:")
	if not tableName or len(tableName.replace(" ","")) == 0 or tableName == "*":
		tableName = input ("Please try again, enter table like:")

	db = MyDatabase()
	selectQuery = "select table_name from INFORMATION_SCHEMA.TABLES where table_name like '%"+tableName+"%';"
	rows = db.getSQLResult(selectQuery)

	for row in rows:
		deleteQuery = "drop table "+row['table_name']+";"
		db.query(deleteQuery)
	db.close()

if __name__ == "__main__":
    main()