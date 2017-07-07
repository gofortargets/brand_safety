import cPickle as pkl
import sys

def main():
	in_file = sys.argv[1]
	print "Loading", in_file
	with open(in_file, "rb") as fp:
	    X_train = pkl.load(fp)
	    y_train = pkl.load(fp)
	    train_classes = pkl.load(fp)
	    vocabulary = pkl.load(fp)
	    mapping = pkl.load(fp)
	out_file = sys.argv[2]
	print "Loading", out_file
	with open(out_file, "wb") as fp:
		pkl.dump(vocabulary, fp)

if __name__ == '__main__':
	main()