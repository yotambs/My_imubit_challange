import numpy as np


class Divid_To_Groups:

    def __init__(self,margin=10):
        self.margin = margin
        self.groups = []

    def create_new_group(self,num):
        new_group = []
        new_group.append(num)
        return new_group

    def enter_new_num(self,num):
        if self.groups == []:
            self.groups.append(self.create_new_group(num))
        else:
            assign_flag = False
            for i,curr_group in enumerate(self.groups):
                if (np.abs(num - np.min(curr_group)) < self.margin) or\
                   (np.abs(num - np.max(curr_group)) < self.margin) or\
                   (num > np.min(curr_group) and num < np.max(curr_group)):
                    assign_flag = True
                    curr_group.append(num)

                    if (i+1) < len(self.groups):
                        if np.abs(np.min(self.groups[i+1]) - np.max(curr_group)) < self.margin:
                            self.groups[i] = self.groups[i] + self.groups[i+1]
                            del (self.groups[i+1])

                    break

            if  assign_flag == False:
                 self.groups.append(self.create_new_group(num))

            self.groups.sort()

    def print_all_gropus(self):
        for i,curr_group in enumerate(self.groups):
            print('group {}:'.format(i))
            print(curr_group)

divide_to_gropus = Divid_To_Groups(9)

divide_to_gropus.enter_new_num(1)
divide_to_gropus.enter_new_num(2)
divide_to_gropus.enter_new_num(3)
divide_to_gropus.enter_new_num(15)
divide_to_gropus.enter_new_num(16)
divide_to_gropus.enter_new_num(17)
divide_to_gropus.enter_new_num(95)
divide_to_gropus.enter_new_num(40)
divide_to_gropus.enter_new_num(8)
divide_to_gropus.enter_new_num(100)
divide_to_gropus.enter_new_num(60)
divide_to_gropus.enter_new_num(52)

divide_to_gropus.print_all_gropus()
