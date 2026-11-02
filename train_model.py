import model_training

result = model_training.train_models()
print(result['best_model_name'])
print(result['results'])
