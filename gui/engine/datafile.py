from gui.engine.abstractdatabase import AbstractDatabase

import json
import os

class DataFile(AbstractDatabase):
    
    # signals
    # userNameChanged = Signal()
    # passwordChanged = Signal()
    # databaseNameChanged = Signal()
    
    
    def __init__(self, file_name = "icsdb", file_version = "1.0", user_name = "admin", password = "654321"):
        self._databaseName = file_name
        self._userName = user_name
        self._password = password
        self._database = []
        self._unsaved = False
        
        self._databaseVersion = file_version
        self._keys = []
        self.init()
    
    
    def user_name(self) -> str:
        return self._userName
    
    
    def set_user_name(self, value : str):
        self._userName = value
        
    def database_name(self) -> str:
        return self._databaseName

    def set_database_name(self, value: str):
        self._databaseName = value
        
    def password(self):
        return self._password
    
    def set_password(self, value : str):
        self._password = value
        
    
    def read_database(self) -> bool:
        self._database = []
        self._keys = []
        
        try:
            with open(self._databaseName, "r+") as rf:
                j = json.load(rf)
                
                # version check
                if (not "file_version" in j) or (not "database" in j):
                    print("Invalid database file. will be removed.")
                    os.remove(self._databaseName)                    
                    return False
                
                if j['file_version'] != "1.0":
                    # recreate captions
                    os.remove(self._databaseName)
                    return False
                
                data = j['database']
                if not isinstance(data, list):
                    print("Invalid database file.")
                    return False
                self._database = data
                for d in self._database:
                    if 'filename' in d:
                        self._keys.append(os.path.normpath(d['filename']))
                return True
        except FileNotFoundError:
            print("The file does not exist.")
            return False
        except json.JSONDecodeError:
            print ("Error decoding JSON.")
            return False
            
            
    def write_database(self):
        j = {
            "file_version": self._databaseVersion,
            "user": self._userName,
            "database": self._database
        }
        try: 
            with open(self._databaseName, "w+") as wf:
                json.dump(j, wf, indent=4)
        except Exception as e:
            print ("Unable to write database file.", e)
            
            
    def init(self) -> bool:
        self.read_database()
    
    def insert(self, **kwargs) -> bool:
        filename = kwargs['filename']
        captions = kwargs['captions']
        filename = os.path.normpath(filename)
        
        print(f"Inserting caption for {filename} into database: {captions}")
        self._database.append({ 'filename': filename, 'captions': captions })
        self._keys.append(filename)
        self._unsaved = True
        print("Row inserted successfully.")
        return True
    
    
    def remove(self, **kwargs) -> bool:
        index = kwargs['index']
        filename = kwargs['filename']
        
        if index is not None:
            self._database.pop(index)
            self._keys.pop(index)
        
        if filename is not None:
            filename = os.path.normpath(filename)
            for item in self._database:
                if isinstance(item, dict) and "filename" in item and item["filename"] == filename:
                    self._database.remove(item)
                    self._keys.remove(filename)
                    break
        
        print ("Row removed successfully.")
        return True
    
    def find(self, **kwargs) -> list:
        filename = kwargs['filename']
        filename = os.path.normpath(filename)
        for item in self._database:
            if isinstance(item, dict) and 'filename' in item and item['filename'] == filename:
                return [item]

        return None
    
    def contains(self, **kwargs) -> bool:
        filename = kwargs['filename']
        filename = os.path.normpath(filename)
        return filename in self._keys
    
    def clear(self, **kwargs):
        self._database = []
        self._keys = []
        
    def close(self):
        if self._unsaved:
            self.write_database()
            
    