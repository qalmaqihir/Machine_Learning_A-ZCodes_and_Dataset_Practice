# print("the {} side {1} {2}".format('bright','of','liofe'))
# x='abcdef'
# i="a"
# while i in x[:-1]:
#     print(i, end=" ")


# l=[1,2,3,4,5]
# m=map(lambda x: 2**x,l)
# print(list(m))

# z=set("abc")
# z.add("san")
# z.update(set(['p','q']))
# print(z)

# import re
# result=re.findall("Welcome to Turing","Welcome",1)
# print(result)

# t='%(a)s %9b)s %(c)s'
# print(t%dict(a='Welcome',b='to',c='Turing'))


# def func1():
#     x=50
#     return x
#
# func1()
# print(x)

# data=[1,2,3]
# def incr(x):
#     return x+1
#
# print(list(map(incr, data)))


# Y=[2,5J,6]
# Y.sort()
# print(Y)


# def f(x,l=[]):
#     for i in range(x):
#         l.append(i*i)
#     return l
#
# print(f(2))
# print(f(3,[3,2,1]))
# print(f(3))
#
# l=[]
# l.insert()

a=[1,2,3,4]
b=[sum(a[0:x+1]) for x in range(0,len(a))]
print(b)
