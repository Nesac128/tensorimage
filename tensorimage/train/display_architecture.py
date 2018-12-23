def display_architecture(**layers):
    for layer_name, shape in zip(layers.keys(), layers.values()):
        print(layer_name+' ', shape)