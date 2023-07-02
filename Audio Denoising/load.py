 # load json and create model
    json_file = open(weights_path+'/'+'model_best.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path+'/'+'model_best'+'.h5')
    print("Loaded model from disk")