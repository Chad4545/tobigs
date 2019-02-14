class Student:
    def __init__(self, id=None, name=None):
        self.id = id
        self.name = name
        
    def __str__(self):
        pass
    
    def __lt__(self, other):
        return len(self) < len(other)

