class Student:
    def __init__(self, num=None, name=None):
        self.num = num
        self.name = name
        
    def __str__(self):
        return "{}번  {}".format(self.num, self.name)
    
    def __lt__(self, other):
        return self.num < other.num

