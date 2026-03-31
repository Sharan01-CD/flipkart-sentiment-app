import onnxruntime as ort
import pickle
import numpy as np

with open('simple_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

word_index = tokenizer['word_index']
text = 'this product is amazing'
seq = [word_index.get(w, 1) for w in text.lower().split()]
print('Sequence:', seq)

padded = np.zeros((1, 50), dtype=np.float32)
padded[0, :len(seq)] = seq
print('Padded:', padded)

session = ort.InferenceSession('lstm_model.onnx')
proba = session.run(None, {'keras_tensor': padded})[0][0]
print('Probabilities:', proba)
print('Predicted:', ['Negative','Neutral','Positive'][int(np.argmax(proba))])
