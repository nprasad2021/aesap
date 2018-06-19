
import parameters

#### PARAMETERS
dsts = ['sap_data', 'all_data', 'open_data']
lrs = [.005, .001, .0005, .0001]
img_sizes = [100, 150, 200, 224]
scale = [1,32,64,128]
slide = [-64, 0, 64]
loss = ['l1', 'l2']


def gen_tune_exp(precursor):

	opt = []
	idx = 0

	# PARAMETERIZE BASELINE
	opt += [parameters.Experiment(identity=idx, precursor=precursor)]

	idx += 1
	for dst in dsts:
		for sl in slide:
			for sc in scale:
				for lr in lrs:
					for imsize in img_sizes:
						for l in loss:

							name = 'tune_' + str(lr) + '_' + dst + '_' + str(imsize) + '_sc' + str(sc) + '_sl' + str(sl) 
							opt += [parameters.Experiment(identity=idx, name=name, precursor=precursor, datatype=dst)]
							opt[-1].image_size = imsize
							opt[-1].learning_rate = lr
							opt[-1].outfile = 'finalresults.txt'
							opt[-1].scale = sc
							opt[-1].slide = sl
							opt[-1].loss = l

							idx += 1

	print('tune experiments ind num:', len(opt))
	return opt

if __name__ == '__main__':
	gen_tune_exp('dfaasdfadsf')