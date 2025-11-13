from PySide2.QtCore import QObject, Property, Signal
from PySide2.QtSql import QSqlDatabase, QSqlQuery

from gui.engine.abstractdatabase import AbstractDatabase

DATABASE_TYPE = "QSQLITE" # "QMYSQL" "QPSQL"

class Database(AbstractDatabase):
    
    def __init__(self, db_name = "icsdb", user_name = "admin", password = "654321", parent : QObject = None):
        
        super().__init__(self, db_name, user_name, password, parent)
    
    def init(self) -> bool:
        self._database = QSqlDatabase.addDatabase("QSQLITE")
        self._database.setHostName("localhost")
        self._database.setDatabaseName(self._databaseName)
        self._database.setUserName(self._userName)
        self._database.setPassword(self._password)
        if self._database.open():
            self.create_caption_table()            
            return True
        else:
            print("Unable to open database.", self._database.lastError().text())
            return False
    
        
    def create_caption_table(self) -> False:
        query = QSqlQuery()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS captions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(256) NOT NULL,
            caption TEXT
        )
        """
        
        if not query.exec_(create_table_query):
            print ("Error creating table:", query.lastError().text())
            return False
        
        print ("Caption table created.")
        return True
    
    
    def insert(self, filename : str, caption : str) -> bool:
        query = QSqlQuery()
        
        inset_row_query = """
        INSERT INTO captions (filename, caption) VALUES (:filename, :caption)
        """
        
        query.prepare(inset_row_query)
        query.bindValue(":filename", filename)
        query.bindValue(":caption", caption)
        
        if not query.exec_():
            print("Error inserting row:", query.lastError().text())
            return False
        
        print("Row inserted successfully.")
        return True
    
    
    def remove(self, id) -> bool:
        if id is None:
            
            return False
        
        query = QSqlQuery()
        
        remove_row_query = """
        DELETE FROM captions WHERE id = :id
        """
        
        query.prepare(remove_row_query)
        query.bindValue(":id", id)
        
        if not query.exec_():
            print("Error removing row:", query.lastError().text())
            return False
        
        print ("Row removed successfully.")
        return True
    
    
    def remove(self, filename = None) -> bool:
        if filename is None:
            return False
        
        query = QSqlQuery()
        
        remove_row_query = """
        DELETE FROM captions WHERE filename = :filename
        """
        
        query.prepare(remove_row_query)
        query.bindValue(":filename", filename)
        
        if not query.exec_():
            print("Error removing row:", query.lastError().text())
            return False
        
        print ("Row removed successfully.")
        return True
    
    
    def close(self):
        self._database.close()
            
    
    # properties
    databaseName = Property(str, database_name, notify=databaseNameChanged)
    userName = Property(str, user_name, notify=userNameChanged)
    password = Property(str, user_password, notify=passwordChanged)