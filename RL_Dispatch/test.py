x = 0
y = 1
print(locals)
print(locals())
print(type(locals()))
locals()['y'] = 2
print(y)