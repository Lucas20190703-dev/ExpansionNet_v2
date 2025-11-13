from abc import ABC, abstractmethod

class AbstractDatabase(ABC):
    
    # virtual methods
    @abstractmethod
    def init(self) -> bool:
        """Init database

        Returns:
            bool: return True if initialization is success
        """
        pass
    
    @abstractmethod
    def insert(self, **kwargs) -> int:
        """insert data into database

        Returns:
            int: return id, -1 if failed
        """
        pass
    
    @abstractmethod
    def remove(self, **kwargs) -> bool:
        """remove data from database with the conditions given in **kwargs

        Returns:
            bool: return True if suceess, otherwise False
        """
        pass
    
    @abstractmethod
    def clear(self, **kwargs):
        """clear all data from database

        """
        pass
    
    @abstractmethod
    def find(self, **kwargs) -> list:
        """get the rows with the conditions given in **kwargs

        Returns:
            list: _description_
        """
        pass
    
    @abstractmethod
    def close(self):
        """close the database
        """
        pass