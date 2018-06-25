
import parameters

#### PARAMETERS
# Final TUNING
dsts = ['all_data', 'open_data', 'sap_data']
lrs = [.001, .0005, .0001]
img_sizes = [144, 192]
scale = [1]
slide = [0]
bs = [32, 64]
build = [3,2,1]
num_units = [50, 100, 150, 200]


def gen_tune_exp(precursor):

	opt = []
	idx = 0

	# PARAMETERIZE BASELINE
	opt += [parameters.Experiment(identity=idx, precursor=precursor)]

	idx += 1
	for dst in dsts:
		for sl in slide:
			for sc in scale:
				for b in bs:
					for lr in lrs:
						for imsize in img_sizes:
							for bill in build:
								for hidden in num_units:


									name = 'accuracy' #+ str(lr) + '_' + dst + '_' + str(imsize) + '_sc' + str(sc) + '_sl' + str(sl) + '_bui' + str(bill) + '_l' + str(l) + '_bs' + str(b)
									opt += [parameters.Experiment(identity=idx, name=name, precursor=precursor, datatype=dst)]
									opt[-1].image_size = imsize
									opt[-1].learning_rate = lr
									opt[-1].outfile = 'finalresults.txt'
									opt[-1].scale = sc
									opt[-1].slide = sl
									opt[-1].batch_size = b
									opt[-1].build = bill
									opt[-1].restart = True
									opt[-1].mode = 'both'
									opt[-1].num_units = hidden
									opt[-1].category = 'arch/'

									idx += 1

	print('tune experiments num:', len(opt))
	return opt

if __name__ == '__main__':
	gen_tune_exp('dfaasdfadsf')