a = 15
_b = 25
def f1():
    """This is the doctsring for the f1 function"""
    print("Function 1 from module")
def f2():
    print("Function 2 from module")
def _f3():
    print("Private Function 3 from module")
def checker():
    print("file1 __name__ = %s" %__name__)
    if __name__ == "__main__":
        print ("file1 is executed directly")
    else:
        print ("file1 is imported")

checker()