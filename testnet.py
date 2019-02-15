from keras.models import load_model
import sys

from testing import run_test

model = load_model(sys.argv[1])

run_test(model, sys.argv[2])
