# This file contains modules written by me to save and load model weights.
# The reason is that I do not use the h5py package used by Keras to save
# Models. I prefer to use the Age old method of Pickle.
#
# Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>

def save_model_weight(model):
  """Since I do not use the H5Py package used by Keras for saving models,
  I wrote this function to store my model using the age old pickle

  Args:
    model: Instance of Sequential Class with model weights

  Returns:
    None. We are not returning anything for now. Later might be added 
  """
  weights = model.get_weights()
  df = pd.DataFrame(weights)
  try:
    df.to_pickle('saved_models/phoc_weights.pkl')
  except:
    print ("Cannot save the model. Most Likely reason is: Out of Space")


def load_model_weight(model):
  """This function will load the model weights from the saved model weights.

  Args:
    model: Instance of Sequential Class i.e. get the network architecture.

  Returns:
    model: Instance of Sequential Class with weights loaded from pickle.
  """
  df = pd.read_pickle('saved_models/phoc_weights.pkl')
  tmp_weights = df.values
  N = len(tmp_weights)
  weights = []
  for i in range(N):
    weights.append(tmp_weights[i][0])
  model.set_weights(weights)
  return model