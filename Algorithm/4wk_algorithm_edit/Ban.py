class Ban:
    def __init__(self, no=None):
        self.no = no
        self.student_list =[]
    
    def __str__(self):
        pass
    
    def __lt__(self, other):
        return len(self) < len(other)
    
    def __eq__(self, other):
        return len(self) == len(other)
    
    def count_student(self):
        """해당 반에 속해있는 학생들의 수 return"""

