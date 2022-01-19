class MyAction:
    def __init__(self, name):
        self.name = name
        self.value = 42
        self.value_error = None

    def set_value_bad(self):
        del self.value
        self.value_error = KeyError

    def __getattr__(self, a_name):
        print('in special thing')
        if self.value_error is not None and a_name == 'value':
            raise self.value_error
        else:
            raise AttributeError


test1 = MyAction('good')

print(test1)
print(test1.name)
print(test1.__getattr__)
print(test1.value)

test1.set_value_bad()
print()

print(test1)
print(test1.name)
print(test1.__getattr__)
print(test1.value)
