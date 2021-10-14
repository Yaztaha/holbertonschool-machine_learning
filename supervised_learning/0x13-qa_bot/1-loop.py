#!/usr/bin/env python3
""" QA loop script """


while 1:
    question = input("Q: ")
    words = ["exit", "quit", "goodbye", "bye"]

    if question.lower().strip() in words:
        print("A: Goodbye")
        exit(0)
    else:
        print("A: ")
