import ctgan

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    discrete_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'income'
    ]
    data = ctgan.load_demo()
    model = ctgan.CTGANSynthesizer(epochs=3, verbose=True)
    model.fit(data, discrete_columns)
    print([val.cpu().numpy() for name, val in model._generator.state_dict().items()
                if 'bn' not in name])

