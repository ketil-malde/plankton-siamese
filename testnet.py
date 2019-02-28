from keras.models import load_model
import sys

import testing as T

model = load_model(sys.argv[1])

#vs = T.get_vectors(model,sys.argv[2])
#for c in vs:
#    print("Class:",c)
#    print(T.find_nearest(vs, vs[c][0], k=3))

# print(T.knn_test(model, rdir=sys.argv[2], tdir=sys.argv[3], k=5))

T.run_test(model, sys.argv[2])
