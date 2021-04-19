#!/usr/bin/env python3
""" Student scores histogram """
import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
number_bins = math.ceil((student_grades.max() - student_grades.min())/10) - 1
bins = range(40, 101, 10)

plt.hist(student_grades, bins=bins, ec="black")
plt.xlim((0, 100))
plt.ylim((0, 30))
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.show()
