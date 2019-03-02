class Ban:
    def __init__(self, no=None):
        self.no = no
        self.student_list = []

    def __str__(self):
        return "<{}반>  {}명".format(self.no, self.count_student())

    def __lt__(self, other):
       if self.no < other.no:
           return True
       else:
           return False

    def __eq__(self, other):
       if isinstance(self, Ban):
           return self.no == other.no
       return False
    
#         """해당 반에 속해있는 학생들의 수 return"""
    def count_student(self):
        return len(self.student_list)