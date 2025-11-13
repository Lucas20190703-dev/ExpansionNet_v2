
class NameFilter:
    def __init__(self, patern = ""):
        self._pattern = patern
        
    def get_pattern(self):
        return self._pattern
    
    def set_pattern(self, pattern):
        self._pattern = pattern


class CaptionFilter:
    def __init__(self, patern = ""):
        self._pattern = patern
        
    def get_pattern(self):
        return self._pattern
    
    def set_pattern(self, pattern):
        self._pattern = pattern
        
        
class DateFilter:
    def __init__(self, start, end):
        self._start = start
        self._end = end
        
    def get_start_date(self):
        return self._start
    
    def set_start_date(self, date):
        self._start = date
        
    def get_end_date(self):
        return self._end
    
    def set_end_date(self, date):
        self._end = date
        