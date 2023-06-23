import tensorflow as tf
class ExportTranslator(tf.Module):
  def __init__(self, translator):
    self.translator = translator

  def __call__(self, sentence):
    (result,
     attention_weights) = self.translator(sentence, max_length=100)
    return result