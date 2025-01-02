class TestClass:
    
    def __init__(self, **kwargs):
        self.return_dict = kwargs.pop('return_dict', True)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        
        
test1 = TestClass(return_dict = False, output_hidden_states = True)
print(test1.output_hidden_states)